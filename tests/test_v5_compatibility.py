"""V5 default dispatch and explicit access to each older implementation."""


def test_v5_default_dispatch_keeps_all_explicit_versions(monkeypatch):
    import run_v9,v2.entry,v3.entry,v4.entry,v5.entry
    observed=[]
    for number,module in [(2,v2.entry),(3,v3.entry),(4,v4.entry),(5,v5.entry)]:
        monkeypatch.setattr(module,'run',lambda args,n=number:observed.append((n,args)))
    run_v9.run(['main'])
    for number in [2,3,4,5]:run_v9.run(['plan','--code-version='+str(number)])
    assert observed==[(5,['main']),(2,['plan']),(3,['plan']),(4,['plan']),(5,['plan'])]
