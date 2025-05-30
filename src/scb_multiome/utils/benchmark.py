import pandas as pd 

def any_peak_overlap(p1:pd.DataFrame, p2:pd.DataFrame, min_overlap=1):
    c1, s1, e1 = p1['c'], p1['s'], p1['e']
    
    res1 = (p2['c'] == c1)
    o_s = s1*(s1>=p2['s']) + p2['s'].mul(s1<p2['s'])
    o_e = e1*(e1<=p2['e']) + p2['e'].mul(e1>p2['e'])
    overlap = o_e - o_s
    # print(o_s, o_e, overlap)
    res = res1 * (overlap>min_overlap)
    return res.any()