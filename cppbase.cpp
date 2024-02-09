#include <bits/stdc++.h>
#define shit ios_base::sync_with_stdio(false);cin.tie(0);cout.tie(0);
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
typedef long long  ll;
using namespace std;
#define all(v) v.begin(), v.end()
#define rall(v) v.rbegin(),v.rend()
#define int long long
typedef tree<int,null_type,less<int>, rb_tree_tag, tree_order_statistics_node_update> indexed_set;
#define size(a) (int)a.size()
#define vint vector<int>
#define pb push_back
#define pint pair<int, int>
#define range(i, s, e) for (int i = s; i < e; ++i)
//bismillah


vint z_function (string& s) {
	int n = size(s);
	vint z (n);
	for (int i=1, l=0, r=0; i<n; ++i) {
		if (i <= r)
			z[i] = min (r-i+1, z[i-l]);
		while (i+z[i] < n && s[z[i]] == s[i+z[i]])
			++z[i];
		if (i+z[i]-1 > r)
			l = i,  r = i+z[i]-1;
	}
	return z;
}


class SegTree{ // zero-indexed, include l and r, v = 1, tl = 0, tr = n - 1
    vint t;
    int n;

    void build(vint& a, int v, int tl, int tr){
        if (tl == tr)
            t[v] = a[tl];
        else {
            int tm = (tl + tr) / 2;
            build (a, v*2, tl, tm);
            build (a, v*2+1, tm+1, tr);
            t[v] = t[v*2] + t[v*2+1];
        }
    }

    public: 
    SegTree(vint& a, int v, int tl, int tr) {
        n = size(a);
        t.resize(4 * n);
        this -> build(a, v, tl, tr);
	}

    int sum (int v, int tl, int tr, int l, int r) {
        if (l > r)
            return 0;
        if (l == tl && r == tr)
            return t[v];
        int tm = (tl + tr) / 2;
        return sum (v*2, tl, tm, l, min(r,tm)) + sum (v*2+1, tm+1, tr, max(l,tm+1), r);
    }

    void update (int v, int tl, int tr, int pos, int new_val) {
        if (tl == tr)
            t[v] = new_val;
        else {
            int tm = (tl + tr) / 2;
            if (pos <= tm)
                update (v*2, tl, tm, pos, new_val);
            else
                update (v*2+1, tm+1, tr, pos, new_val);
            t[v] = t[v*2] + t[v*2+1];
        }
    }
};


class DSU{
    unordered_map<int, int> parent;
    unordered_map<int, int> rank;
    unordered_map<int, int> size;
    public:
    DSU();
    void make_set(int v);
    void union_sets(int v, int u);
    int find_set(int v);
};

void DSU::make_set(int v){
    parent[v] = v;
	rank[v] = 0;
}

int DSU::find_set(int v){
    if (parent[v] == v){
        return v;
    }
    return parent[v] = find_set(parent[v]);
}

void DSU::union_sets(int a, int b){
    a = this -> find_set(a);
    b = this -> find_set(b);
    if (a != b){
        if (rank[a] < rank[b])
            swap(a, b);
        parent[b] = a;
        if (rank[a] == rank[b])
            ++rank[a];
    }
}


vector<string> split(string& s){
    vector<string> ans;
    string cur = "";
    for (char& chr : s){
        if (chr == '\0')
            return ans;
        else if(chr == '\n' || chr == '\t' || chr == ' '){
            if (cur != "")
                ans.pb(cur);
            cur = "";
        }else {
            cur += chr;
        }
    }
    if (cur != "")
        ans.pb(cur);
    return ans;
}


int sti(string& s){
    int res = 0;
    bool f = false;
    if (s[0] == '-'){
        f = true;
        s = s.substr(1, size(s));
    }
    for (int i = 0 ; i < size(s) ; ++i){
        res += (s[i] - 48)  * pow(10, size(s) - i - 1);
    }
    if (f)
        res *= -1;
    return res;
}

//
// struct TrieNode{
//     char value;
//     vector<TrieNode> next;
// }
//
// class Trie{
//
// }

template <typename T> void print_arr(T& arr){
    for (auto& i : arr){
        cout << i << ' ';
    }
    cout << '\n';
}

void read_ints(vint& arr, int n){
    for (int i = 0 ; i < n; ++i){
        cin >> arr[i];
    }
}


void solve(){
    return;
}


int32_t main(){
    int t;
    cin >> t;
    while (t--){
        solve();
    }
}
