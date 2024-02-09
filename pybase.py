#!/usr/bin/python3
import sys
from collections import defaultdict as dict, deque
import math
from abc import ABC, abstractmethod


if len(sys.argv) > 1 and sys.argv[1]  in ('-d'):
    sys.stdout = open('out', 'w')

sys.setrecursionlimit(10 ** 5)
input = sys.stdin.readline
inf = float('inf')
true = True
false = False

class DSU:
    def __init__(self):
        self.parent = dict(int)
        self.rank = dict(int)
        self.size = dict(int)

    def make_set(self, v):
        self.parent[v] = v
        self.rank[v] = 0

    def find_set(self, v):
        if self.parent[v] == v:
            return v
        self.parent[v] = self.find_set(self.parent[v])
        return self.parent[v]

    def union_sets(self, a, b):
        a = self.find_set(a)
        b = self.find_set(b)
        if a != b:
            if self.rank[a] < self.rank[b]:
                a, b = b, a
            self.parent[b] = a
            if self.rank[a] == self.rank[b]:
                self.rank[a] += 1


def lcs(s1, s2):
    n1, n2 = len(s1), len(s2)
    dp = [[0 for _ in range(n1)] for _ in range(n2)]
    for i in range(n1):
        dp[0][i] = dp[0][i - 1]
        if s1[i] == s2[0]:
            dp[0][i] = 1
    for i in range(n2):
        dp[i][0] = dp[i - 1][0]
        if s1[0] == s2[i]:
            dp[i][0] = 1
    for i in range(1, n2):
        for j in range(1, n1):
            if s1[j] == s2[i]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


class Heap:
    def __init__(self):
        self.heap = []
        self.size = 0

    def push(self, value):
        self.size += 1
        self.heap.append(value)
        pos = len(self.heap) - 1
        while value > self.heap[(pos - 1) // 2] and (pos - 1) // 2 >= 0:
            self.heap[(pos - 1) // 2], self.heap[pos] = self.heap[pos], self.heap[(pos - 1) // 2]
            pos = (pos - 1) // 2
            if pos == 0:
                break

    def pop(self):
        self.size -= 1
        self.heap[0], self.heap[-1] = self.heap[-1], self.heap[0]
        res = self.heap.pop()
        pos = 0
        while (2 * pos + 1 < len(self.heap) and \
                self.heap[pos] < self.heap[2 * pos + 1]) or \
                (2 * pos + 2 < len(self.heap) and self.heap[pos] < self.heap[2 * pos + 2]):
            maxson = 2 * pos + 1
            if 2 * pos + 2 < len(self.heap):
                if self.heap[2 * pos + 1] < self.heap[2 * pos + 2]:
                    maxson  = 2 * pos + 2
            self.heap[pos], self.heap[maxson] = self.heap[maxson], self.heap[pos]
            pos = maxson
        return res

    def top(self):
        return self.heap[0]

    def __len__(self):
        return self.size
    
    def __str__(self):
        return str(self.heap)


class Graph(ABC):
    def __init__(self, n: int = 0, *args, **kwargs):
        if not args and not kwargs:
            self.gr = dict(list)
        elif args:
            self.gr = args
        else:
            self.kwargs
    
    @abstractmethod
    def push_edge(self, u: int, v: int) -> None:
        ...

    def remove_loop_and_parallels(self) -> None:
        for v in self.gr:
            self.gr[v] = list(set(self.gr[v]))


    @abstractmethod
    def min_dist(self, u: int, v: int) -> int:
        ...

    @abstractmethod
    def is_bipartite(self) -> bool:
        ...
    
    @abstractmethod
    def count_components(self) -> int:
        ...

    @abstractmethod
    def is_cyclic(self) -> bool:
        ...
        
    def __iter__(self):
        for v in self.gr[v]:
            yield v

    def __str__(self) -> str:
        return str(self.gr)


class UnweightedUndirectedGr(Graph):

    def push_edge(self, u: int, v: int) -> None:
         self.gr[u].append(v)
         self.gr[v].append(u)

    def min_dist(self, u: int, v: int) -> None:
        q = deque()
        q.append((u, 0))
        visited = dict(lambda: False)
        
        while q:
            vert, dist = q.popleft()
            visited[vert] = True
            if vert == v:
                return dist
            for neigh in self.gr[vert]:
                if not visited[neigh]:
                    q.append((neigh, dist + 1))

    def is_bipartite(self) -> bool:
        color = 1
        visited = dict(lambda : 0)
        for v in self.gr:
            if not visited[v]:
                st = deque()
                st.append((v, color))
                while st:
                    vert, color = st.pop()
                    visited[vert] = color
                    for neigh in self.gr[vert]:
                        if visited[neigh]:
                            if visited[neigh] == color:
                                return False
                        else:
                            st.append((neigh, 3 - color))
        return True
    
    def count_components(self) -> int:
        cmp = 0
        visited = dict(lambda : 0)
        for v in self.gr:
            if not visited[v]:
                cmp += 1
                st = deque()
                st.append(v)
                while st:
                    vert = st.pop()
                    visited[vert] = cmp
                    for neigh in self.gr[vert]:
                        if not visited[neigh]:
                            st.append(neigh)
        return cmp


class UnweightedDirectedGr(UnweightedUndirectedGr):
    
    def push_edge(self, u: int, v: int) -> None:
         self.gr[u].append(v)
         if v not in self.gr:
             self.gr[v] = []

    def min_dist(self, u: int, v: int) -> None:
        q = deque()
        q.append((u, 0))
        visited = dict(lambda: False)
        
        while q:
            vert, dist = q.popleft()
            visited[vert] = True
            if vert == v:
                return dist
            for neigh in self.gr[vert]:
                if not visited[neigh]:
                    q.append((neigh, dist + 1))

        return inf
    # TODO
    def topsort(self) -> list[int]:
        ans = []
        visited = dict(lambda : False)
        for v in self.gr:
            if not visited[v]:
                st = deque()
                st.append(v)
                while st:
                    vert = st.pop()
                    for neigh in self.gr[vert]:
                        if not visited[neigh]:
                            st.append(neigh)
                    visited[vert] = true
                    ans.append(vert)
        return ans[::-1] 
    # TODO doesnt wors as expected 
    def is_cyclic(self) -> bool:
        visited = dict(lambda :0)
        for v in self.gr:
            if not visited[v]:
                st = deque()
                st.append((v, 1, v))
                while st:
                    vert, color, from_ = st.pop()
                    visited[vert] = 1
                    for neigh in self.gr[vert]:
                        if neigh == from_:
                            continue
                        if not visited[neigh]:
                            st.append((neigh, 1, vert))
                        else:
                            if visited[neigh] == 1:
                                return True
                    visited[vert] = 2
        return true


def lca(u, v, parents):
    h1 = find_depth(u)
    h2 = find_depth(v)
    while h1 != h2:
        if h1 > h2:
            u = parents[u]
            h1 -= 1
        else:
            v = parents[v]
            h2 -= 1
    while v != u:
        u = parents[u]
        v = parents[v]
    return u


def bs_in_arr(arr, l, r, check):
    while l <= r:
        mid = (l + r) >> 1
        if check(arr[mid]):
            l = mid + 1
        else:
            r = mid - 1
    return l


def bs_flat(l, r, check):
    while l <= r:
        mid = (l + r) >> 1
        if check(mid):
            l = mid + 1
        else:
            r = mid - 1
    return l

    
def read_ints() -> map:
    return map(int, input().split())


def solve() -> None:
    ...


def main():
    t = int(input()) 
    for _ in range(t):
        solve()


if __name__ == '__main__':
    main()
