// Sokoban solver: push-optimal, room by room.
//
// Strategy (see PLAN.md in the bounty workspace): a generated level is a
// one-square hallway with 1-4 rooms off it; a box pushed out of a room can
// never be pushed back, so every box stays in its room and the level's
// push-optimum is the sum of its rooms' optima. Each room (3-4 boxes in a
// 6x6 slot) is solved push-optimally with a breadth-first search over pushes,
// pruned by dead squares (reverse pull-back from the goals) and frozen 2x2
// blocks; the player's walks between pushes are shortest paths. A* (same
// admissible matching heuristic) and greedy best-first are fallbacks when a
// room's BFS exhausts its node budget, so a valid solution is still written.
//
// Deterministic: fixed direction order, sorted boxes, FIFO/priority queues
// with total tie-breaks, budgets by node count. No clock, no randomness,
// no reads outside /input and /output.
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const IDS: string[] = ['i001', 'i002', 'i003', 'i004', 'i005'];
const MAX_SOLUTION_BYTES = 65_536;
const BFS_BUDGET = 250_000;
const ASTAR_BUDGET = 250_000;
const GREEDY_BUDGET = 100_000;

// Directions in fixed order: [dx, dy, letter]. Every iteration in this file
// follows this order, which is part of the determinism contract.
const DIRS: Array<[number, number, string]> = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];

export interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

export interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
}

export interface RoomSolution {
  pushes: number;
  moves: string;
  states: number;
  method: string;
}

export interface RoomDetail {
  room: number;
  boxes: number;
  states: number;
  method: string;
  pushes: number;
}

export interface LevelSolution {
  moves: string;
  pushes: number;
  rooms: RoomDetail[];
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/** Parse XSB text. Lenient on purpose: unknown chars are floor, ragged rows
 *  and the outer frame are walls, a blank line in the middle is a wall row,
 *  CRLF is stripped before measuring, and a missing player falls back to the
 *  first floor square. Never throws on malformed text. */
export function parseLevel(text: string): Level {
  const raw: string[] = text.split('\n').map((r) => (r.endsWith('\r') ? r.slice(0, -1) : r));
  while (raw.length > 0 && raw[raw.length - 1] === '') raw.pop();
  const h = raw.length;
  let w = 0;
  for (const r of raw) if (r.length > w) w = r.length;
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  for (let y = 0; y < h; y++) {
    const row = raw[y];
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  for (let x = 0; x < w; x++) {
    wall[x] = 1;
    wall[(h - 1) * w + x] = 1;
  }
  for (let y = 0; y < h; y++) {
    wall[y * w] = 1;
    wall[y * w + (w - 1)] = 1;
  }
  if (player < 0) {
    for (let i = 0; i < wall.length; i++) {
      if (!wall[i]) {
        player = i;
        break;
      }
    }
  }
  goals.sort((a, b) => a - b);
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// ---------------------------------------------------------------------------
// Dead squares (live squares)
// ---------------------------------------------------------------------------

/** Squares from which a box alone can be pushed onto some goal, found by
 *  pulling back from the goals: a box on y can be pushed to a live square x
 *  when the player can stand on y - d. A box on a dead square can never
 *  reach a goal, so any state holding one is pruned. */
export function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue: number[] = [];
  for (const g of goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (let k = 0; k < 4; k++) {
      const d = DIRS[k][1] * w + DIRS[k][0];
      const y = x - d;
      const behind = y - d;
      if (y < 0 || y >= wall.length || behind < 0 || behind >= wall.length) continue;
      if (!wall[y] && !wall[behind] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

// ---------------------------------------------------------------------------
// Rooms
// ---------------------------------------------------------------------------

/** Split the level into rooms: cut the dead floor squares of the player's
 *  row (the hallway) out of the floor; each connected part left that holds
 *  a box is a room. Null when the level is no set of rooms (a box in the
 *  hallway, a goal in the cut, boxes != goals in a part, or a box-part that
 *  also holds a hallway square, which would mean a live hallway square). */
export function roomsOf(level: Level, live: Uint8Array): Room[] | null {
  const { w, h, wall, goals, boxes, player } = level;
  if (player < 0 || w === 0 || h === 0) return null;
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  for (const b of boxes) if (cut[b]) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares: number[] = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i];
      const px = p % w;
      if (p >= w) {
        const q = p - w;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
      if (p + w < wall.length) {
        const q = p + w;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
      if (px > 0) {
        const q = p - 1;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
      if (px < w - 1) {
        const q = p + 1;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) {
    const r = part[b];
    if (r < 0) return null;
    rooms[r].boxes.push(b);
  }
  for (const g of goals) {
    const r = part[g];
    if (r < 0) return null;
    rooms[r].goals.push(g);
  }
  const out: Room[] = [];
  for (const r of rooms) {
    if (r.boxes.length === 0) {
      if (r.goals.length !== 0) return null;
      continue;
    }
    if (r.boxes.length !== r.goals.length) return null;
    for (const s of r.squares) if (Math.floor(s / w) === row) return null;
    out.push(r);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Small search helpers
// ---------------------------------------------------------------------------

/** Canonical key of a state: the lowest square of the player's reachable
 *  region, then the sorted box squares, as digits in base `size`. Exact
 *  while size ** (boxes + 1) < 2 ** 53 (rooms: <= 5 boxes on <= 255 squares). */
export function stateKey(boxes: number[], low: number, size: number): number {
  let k = low;
  for (const b of boxes) k = k * size + b;
  return k;
}

/** Flood fill from `from` around walls and boxes, marking `seen` with
 *  `stamp`; returns the lowest square reached. */
function flood(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stack: Int32Array,
  stamp: number,
): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    const px = p % w;
    if (p >= w) {
      const q = p - w;
      if (!wall[q] && !occupied[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
    if (p + w < wall.length) {
      const q = p + w;
      if (!wall[q] && !occupied[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
    if (px > 0) {
      const q = p - 1;
      if (!wall[q] && !occupied[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
    if (px < w - 1) {
      const q = p + 1;
      if (!wall[q] && !occupied[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Shortest walk from `from` to `to` around walls and boxes, lower-case
 *  letters, ties broken by the fixed direction order; null if unreachable. */
export function walkTo(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  to: number,
): string | null {
  if (from === to) return '';
  const size = wall.length;
  const prev = new Int32Array(size).fill(-1);
  const how = new Uint8Array(size);
  const queue: number[] = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    const px = p % w;
    for (let k = 0; k < 4; k++) {
      const dx = DIRS[k][0];
      const dy = DIRS[k][1];
      if (dx < 0 && px === 0) continue;
      if (dx > 0 && px === w - 1) continue;
      if (dy < 0 && p < w) continue;
      if (dy > 0 && p + w >= size) continue;
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = k;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = DIRS[how[p]][2] + s;
  return s;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block: all four
 *  squares walls or boxes, with a box among them off its goal. Only the
 *  blocks holding `t` can have changed, so only those are checked. */
export function frozenAfterPush(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  goal: Uint8Array,
  t: number,
): boolean {
  const h = Math.floor(wall.length / w);
  const tx = t % w;
  const ty = (t - tx) / w;
  for (let oy = -1; oy <= 0; oy++) {
    for (let ox = -1; ox <= 0; ox++) {
      const ax = tx + ox;
      const ay = ty + oy;
      if (ax < 0 || ay < 0 || ax + 1 >= w || ay + 1 >= h) continue;
      const a = ay * w + ax;
      const cells = [a, a + 1, a + w, a + w + 1];
      let blocked = true;
      let offGoalBox = false;
      for (const c of cells) {
        if (!wall[c] && !occupied[c]) {
          blocked = false;
          break;
        }
        if (occupied[c] && !goal[c]) offGoalBox = true;
      }
      if (blocked && offGoalBox) return true;
    }
  }
  return false;
}

/** The square the player stands on after `moves` from `from` (moves are
 *  legal by construction, so no wall checks are needed). */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) {
    if (ch === 'u') p -= w;
    else if (ch === 'd') p += w;
    else if (ch === 'l') p -= 1;
    else if (ch === 'r') p += 1;
  }
  return p;
}

// ---------------------------------------------------------------------------
// Push search over one sub-problem (a room, or the whole level as fallback)
// ---------------------------------------------------------------------------

interface SearchCtx {
  w: number;
  wall: Uint8Array;
  goal: Uint8Array;
  goals: number[];
  live: Uint8Array;
  size: number;
  seenA: Int32Array;
  seenB: Int32Array;
  stack: Int32Array;
  stampA: number;
  stampB: number;
  occupied: Uint8Array;
}

function makeCtx(w: number, wall: Uint8Array, goals: number[], live: Uint8Array): SearchCtx {
  const goal = new Uint8Array(wall.length);
  for (const g of goals) goal[g] = 1;
  return {
    w,
    wall,
    goal,
    goals,
    live,
    size: wall.length,
    seenA: new Int32Array(wall.length),
    seenB: new Int32Array(wall.length),
    stack: new Int32Array(wall.length),
    stampA: 0,
    stampB: 0,
    occupied: new Uint8Array(wall.length),
  };
}

function isSolved(boxes: number[], goal: Uint8Array): boolean {
  for (const b of boxes) if (!goal[b]) return false;
  return true;
}

/** Reconstruct the move string of a found state from the parent arrays. */
function reconstruct(
  ctx: SearchCtx,
  statesBoxes: number[][],
  statesPlayer: number[],
  parent: number[],
  pushFrom: number[],
  pushDir: number[],
  found: number,
  startPlayer: number,
  startBoxes: number[],
): string {
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = startPlayer;
  let boxes = startBoxes;
  for (const s of path) {
    const nextBoxes = statesBoxes[s];
    const from = pushFrom[s];
    const d = DIRS[pushDir[s]][1] * ctx.w + DIRS[pushDir[s]][0];
    for (const b of boxes) ctx.occupied[b] = 1;
    const walk = walkTo(ctx.w, ctx.wall, ctx.occupied, player, from - d);
    for (const b of boxes) ctx.occupied[b] = 0;
    if (walk === null) throw new Error('walkTo: push square out of reach');
    moves += walk;
    moves += DIRS[pushDir[s]][2];
    player = from;
    boxes = nextBoxes;
  }
  return moves;
}

/**
 * Breadth-first search over pushes. The first solved state GENERATED is
 * push-optimal (uniform push cost, generation in depth order), so the goal
 * is checked on generation, not on dequeue — the state budget the generator
 * guarantees is calibrated on that count. Returns null on budget exhaustion
 * or when no solution exists.
 */
export function solveRoom(
  w: number,
  wall: Uint8Array,
  goals: number[],
  boxes: number[],
  player: number,
  live: Uint8Array,
  budget: number = BFS_BUDGET,
): RoomSolution | null {
  if (boxes.length === 0) return { pushes: 0, moves: '', states: 0, method: 'bfs' };
  if (boxes.length > goals.length) return null;
  for (const b of boxes) if (!live[b]) return null;
  const ctx = makeCtx(w, wall, goals, live);
  const startBoxes = boxes.slice().sort((a, b) => a - b);
  if (isSolved(startBoxes, ctx.goal)) return { pushes: 0, moves: '', states: 0, method: 'bfs' };
  for (const b of startBoxes) ctx.occupied[b] = 1;
  const statesBoxes: number[][] = [startBoxes];
  const statesPlayer: number[] = [player];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushDir: number[] = [-1];
  const index = new Map<number, number>();
  ctx.stampB++;
  const low0 = flood(w, wall, ctx.occupied, player, ctx.seenB, ctx.stack, ctx.stampB);
  index.set(stateKey(startBoxes, low0, ctx.size), 0);
  let found = -1;
  for (let head = 0; found < 0 && head < statesBoxes.length; head++) {
    const cur = statesBoxes[head];
    for (const b of cur) ctx.occupied[b] = 1;
    ctx.stampA++;
    const here = ctx.stampA;
    flood(w, wall, ctx.occupied, statesPlayer[head], ctx.seenA, ctx.stack, here);
    for (let bi = 0; bi < cur.length && found < 0; bi++) {
      const b = cur[bi];
      for (let k = 0; k < 4; k++) {
        const d = DIRS[k][1] * w + DIRS[k][0];
        const t = b + d;
        if (wall[t] || ctx.occupied[t] || !live[t]) continue;
        if (ctx.seenA[b - d] !== here) continue;
        ctx.occupied[b] = 0;
        ctx.occupied[t] = 1;
        if (frozenAfterPush(w, wall, ctx.occupied, ctx.goal, t)) {
          ctx.occupied[t] = 0;
          ctx.occupied[b] = 1;
          continue;
        }
        const next = cur.slice();
        next[bi] = t;
        next.sort((a, b2) => a - b2);
        ctx.stampB++;
        const low = flood(w, wall, ctx.occupied, b, ctx.seenB, ctx.stack, ctx.stampB);
        const key = stateKey(next, low, ctx.size);
        ctx.occupied[t] = 0;
        ctx.occupied[b] = 1;
        if (index.has(key)) continue;
        if (statesBoxes.length >= budget) {
          for (const x of cur) ctx.occupied[x] = 0;
          return null;
        }
        index.set(key, statesBoxes.length);
        statesBoxes.push(next);
        statesPlayer.push(b);
        parent.push(head);
        pushFrom.push(b);
        pushDir.push(k);
        if (isSolved(next, ctx.goal)) {
          found = statesBoxes.length - 1;
          break;
        }
      }
    }
    for (const x of cur) ctx.occupied[x] = 0;
  }
  if (found < 0) return null;
  const moves = reconstruct(ctx, statesBoxes, statesPlayer, parent, pushFrom, pushDir, found, player, startBoxes);
  return { pushes: countPushes(parent, found), moves, states: statesBoxes.length, method: 'bfs' };
}

function countPushes(parent: number[], found: number): number {
  let n = 0;
  for (let s = found; s > 0; s = parent[s]) n++;
  return n;
}

// --- A* / greedy ------------------------------------------------------------

/** Minimum pushes for one box alone to reach each goal (pull-back BFS per
 *  goal, ignoring the other boxes). Computed once per search. */
function loneDistances(ctx: SearchCtx): Int32Array[] {
  const out: Int32Array[] = [];
  for (const g of ctx.goals) {
    const dist = new Int32Array(ctx.size).fill(-1);
    dist[g] = 0;
    const queue: number[] = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const x = queue[qi];
      for (let k = 0; k < 4; k++) {
        const d = DIRS[k][1] * ctx.w + DIRS[k][0];
        const y = x - d;
        const behind = y - d;
        if (y < 0 || y >= ctx.size || behind < 0 || behind >= ctx.size) continue;
        if (!ctx.wall[y] && !ctx.wall[behind] && dist[y] < 0) {
          dist[y] = dist[x] + 1;
          queue.push(y);
        }
      }
    }
    out.push(dist);
  }
  return out;
}

/** Admissible heuristic: the cheapest assignment of boxes to goals under
 *  the lone-box distances (exact by permutation when boxes == goals <= 5,
 *  else the sum of per-box minima, which is still a lower bound). */
function matchingHeuristic(boxes: number[], goals: number[], dist: Int32Array[]): number {
  if (boxes.length === goals.length && boxes.length <= 5) {
    const perm: number[] = goals.map((_, i) => i);
    let best = Infinity;
    const rec = (i: number, cost: number): void => {
      if (cost >= best) return;
      if (i === boxes.length) {
        best = cost;
        return;
      }
      for (let j = i; j < perm.length; j++) {
        const tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
        const d = dist[perm[i]][boxes[i]];
        if (d >= 0) rec(i + 1, cost + d);
        const tmp2 = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp2;
      }
    };
    rec(0, 0);
    return best;
  }
  let sum = 0;
  for (const b of boxes) {
    let m = Infinity;
    for (let gi = 0; gi < goals.length; gi++) {
      const d = dist[gi][b];
      if (d >= 0 && d < m) m = d;
    }
    if (m === Infinity) return Infinity;
    sum += m;
  }
  return sum;
}

interface HeapEntry {
  p1: number;
  p2: number;
  key: number;
  idx: number;
}

function heapLess(a: HeapEntry, b: HeapEntry): boolean {
  if (a.p1 !== b.p1) return a.p1 < b.p1;
  if (a.p2 !== b.p2) return a.p2 < b.p2;
  return a.key < b.key;
}

function heapPush(heap: HeapEntry[], e: HeapEntry): void {
  heap.push(e);
  let i = heap.length - 1;
  while (i > 0) {
    const p = (i - 1) >> 1;
    if (!heapLess(heap[i], heap[p])) break;
    const tmp = heap[i];
    heap[i] = heap[p];
    heap[p] = tmp;
    i = p;
  }
}

function heapPop(heap: HeapEntry[]): HeapEntry {
  const top = heap[0];
  const last = heap.pop() as HeapEntry;
  if (heap.length > 0) {
    heap[0] = last;
    let i = 0;
    for (;;) {
      const l = 2 * i + 1;
      const r = l + 1;
      let m = i;
      if (l < heap.length && heapLess(heap[l], heap[m])) m = l;
      if (r < heap.length && heapLess(heap[r], heap[m])) m = r;
      if (m === i) break;
      const tmp = heap[i];
      heap[i] = heap[m];
      heap[m] = tmp;
      i = m;
    }
  }
  return top;
}

/**
 * A* (mode 'astar', f = g + h, still push-optimal when it finishes inside
 * its budget) or greedy best-first (mode 'greedy', f = h). Same state model
 * and pruning as the BFS. States are reopened when a better g is found, so
 * A* stays optimal for the admissible matching heuristic. The goal is
 * checked on pop here (a generated goal is not necessarily optimal under f).
 */
function searchHeuristic(
  w: number,
  wall: Uint8Array,
  goals: number[],
  boxes: number[],
  player: number,
  live: Uint8Array,
  budget: number,
  mode: 'astar' | 'greedy',
): RoomSolution | null {
  if (boxes.length === 0) return { pushes: 0, moves: '', states: 0, method: mode };
  if (boxes.length > goals.length) return null;
  for (const b of boxes) if (!live[b]) return null;
  const ctx = makeCtx(w, wall, goals, live);
  const dist = loneDistances(ctx);
  const startBoxes = boxes.slice().sort((a, b) => a - b);
  if (isSolved(startBoxes, ctx.goal)) return { pushes: 0, moves: '', states: 0, method: mode };
  const h0 = matchingHeuristic(startBoxes, goals, dist);
  if (h0 === Infinity) return null;
  for (const b of startBoxes) ctx.occupied[b] = 1;
  const statesBoxes: number[][] = [startBoxes];
  const statesPlayer: number[] = [player];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushDir: number[] = [-1];
  const gScore: number[] = [0];
  const bestG = new Map<number, number>();
  ctx.stampB++;
  const low0 = flood(w, wall, ctx.occupied, player, ctx.seenB, ctx.stack, ctx.stampB);
  const key0 = stateKey(startBoxes, low0, ctx.size);
  bestG.set(key0, 0);
  const heap: HeapEntry[] = [];
  heapPush(heap, mode === 'astar' ? { p1: h0, p2: h0, key: key0, idx: 0 } : { p1: h0, p2: 0, key: key0, idx: 0 });
  let found = -1;
  while (heap.length > 0 && found < 0) {
    const entry = heapPop(heap);
    const head = entry.idx;
    const curG = gScore[head];
    const cur = statesBoxes[head];
    if (curG > (bestG.get(entry.key) as number)) continue; // stale entry
    if (isSolved(cur, ctx.goal)) {
      found = head;
      break;
    }
    for (const b of cur) ctx.occupied[b] = 1;
    ctx.stampA++;
    const here = ctx.stampA;
    flood(w, wall, ctx.occupied, statesPlayer[head], ctx.seenA, ctx.stack, here);
    for (let bi = 0; bi < cur.length; bi++) {
      const b = cur[bi];
      for (let k = 0; k < 4; k++) {
        const d = DIRS[k][1] * w + DIRS[k][0];
        const t = b + d;
        if (wall[t] || ctx.occupied[t] || !live[t]) continue;
        if (ctx.seenA[b - d] !== here) continue;
        ctx.occupied[b] = 0;
        ctx.occupied[t] = 1;
        if (frozenAfterPush(w, wall, ctx.occupied, ctx.goal, t)) {
          ctx.occupied[t] = 0;
          ctx.occupied[b] = 1;
          continue;
        }
        const next = cur.slice();
        next[bi] = t;
        next.sort((a, b2) => a - b2);
        const h = matchingHeuristic(next, goals, dist);
        ctx.stampB++;
        const low = flood(w, wall, ctx.occupied, b, ctx.seenB, ctx.stack, ctx.stampB);
        const key = stateKey(next, low, ctx.size);
        ctx.occupied[t] = 0;
        ctx.occupied[b] = 1;
        if (h === Infinity) continue;
        const ng = curG + 1;
        const prevBest = bestG.get(key);
        if (prevBest !== undefined && prevBest <= ng) continue;
        if (statesBoxes.length >= budget) return null;
        // A known state reached with a better g is reopened as a fresh node
        // (its old node stays in the arrays, unreachable from the new
        // parent chain); an unknown state is stored for the first time.
        bestG.set(key, ng);
        statesBoxes.push(next);
        statesPlayer.push(b);
        parent.push(head);
        pushFrom.push(b);
        pushDir.push(k);
        gScore.push(ng);
        const idx = statesBoxes.length - 1;
        heapPush(heap, mode === 'astar' ? { p1: ng + h, p2: h, key, idx } : { p1: h, p2: ng, key, idx });
      }
    }
    for (const x of cur) ctx.occupied[x] = 0;
  }
  if (found < 0) return null;
  const moves = reconstruct(ctx, statesBoxes, statesPlayer, parent, pushFrom, pushDir, found, player, startBoxes);
  return { pushes: countPushes(parent, found), moves, states: statesBoxes.length, method: mode };
}

/** BFS, then A*, then greedy, per sub-problem. */
function solveRoomWithFallbacks(
  w: number,
  wall: Uint8Array,
  goals: number[],
  boxes: number[],
  player: number,
  live: Uint8Array,
): RoomSolution | null {
  const bfs = solveRoom(w, wall, goals, boxes, player, live, BFS_BUDGET);
  if (bfs !== null) return bfs;
  const astar = searchHeuristic(w, wall, goals, boxes, player, live, ASTAR_BUDGET, 'astar');
  if (astar !== null) return astar;
  return searchHeuristic(w, wall, goals, boxes, player, live, GREEDY_BUDGET, 'greedy');
}

// ---------------------------------------------------------------------------
// Whole level
// ---------------------------------------------------------------------------

export function solveLevelDetail(level: Level): LevelSolution {
  const empty: LevelSolution = { moves: '', pushes: 0, rooms: [] };
  if (level.w === 0 || level.h === 0 || level.player < 0) return empty;
  if (level.boxes.length === 0) return empty;
  if (level.boxes.length > level.goals.length) return empty;
  const live = liveSquares(level);
  for (const b of level.boxes) if (!live[b]) return empty;
  const rooms = roomsOf(level, live);
  if (rooms !== null) {
    let moves = '';
    let pushes = 0;
    let player = level.player;
    const details: RoomDetail[] = [];
    for (let ri = 0; ri < rooms.length; ri++) {
      const room = rooms[ri];
      const wall = level.wall.slice();
      for (let oj = 0; oj < rooms.length; oj++) {
        if (oj === ri) continue;
        const other = rooms[oj];
        const src = oj < ri ? other.goals : other.boxes;
        for (const b of src) wall[b] = 1;
      }
      const solved = solveRoomWithFallbacks(level.w, wall, room.goals, room.boxes, player, live);
      if (solved === null) return empty;
      moves += solved.moves;
      pushes += solved.pushes;
      details.push({ room: ri, boxes: room.boxes.length, states: solved.states, method: solved.method, pushes: solved.pushes });
      player = endOf(level.w, player, solved.moves);
    }
    return { moves, pushes, rooms: details };
  }
  // No room structure: only a small level gets a whole-level search.
  if (level.boxes.length <= 4) {
    const solved = solveRoomWithFallbacks(level.w, level.wall.slice(), level.goals, level.boxes, level.player, live);
    if (solved === null) return empty;
    return {
      moves: solved.moves,
      pushes: solved.pushes,
      rooms: [{ room: 0, boxes: level.boxes.length, states: solved.states, method: `global-${solved.method}`, pushes: solved.pushes }],
    };
  }
  return empty;
}

export function solveLevel(level: Level): string {
  return solveLevelDetail(level).moves;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
  for (const id of IDS) {
    let moves = '';
    try {
      const text = readFileSync(join(inputDir, id, 'level.txt'), 'utf8');
      moves = solveLevel(parseLevel(text));
    } catch {
      moves = '';
    }
    if (moves.length > MAX_SOLUTION_BYTES) moves = moves.slice(0, MAX_SOLUTION_BYTES);
    try {
      mkdirSync(join(outputDir, id), { recursive: true });
      writeFileSync(join(outputDir, id, 'solution.txt'), moves);
    } catch {
      // Nothing more can be done for this instance; continue with the rest.
    }
  }
}

if (resolve(process.argv[1] ?? '') === fileURLToPath(import.meta.url)) main();
