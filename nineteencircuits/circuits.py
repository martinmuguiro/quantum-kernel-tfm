"""
Circuit configurations used in Experiment 1.
"""

CIRCUITS = [
    "rx,rz",
    "rx,rz,cnot-nn-inv",
    "rx,rz,crz-nn-inv",
    "rx,rz,crx-nn-inv",
    "rx,rz,crz-all,rx,rz",
    "rx,rz,crx-all,rx,rz",
    "rx,rz,crz-10,crz-32,rx,rz,crz-21",
    "rx,rz,crx-10,crx-32,rx,rz,crx-21",
    "h,cz-nn-inv,rx",
    "ry,cz-nn-inv,cz-03,ry",
    "ry,rz,cnot-10,cnot-32,ry-1,ry-2,rz-1,rz-2,cnot-21",
    "ry,rz,cz-10,cz-32,ry-1,ry-2,rz-1,rz-2,cz-21",
    "ry,crz-30,crz-23,crz-12,crz-01,ry,crz-32,crz-03,crz-10,crz-21",
    "ry,crx-30,crx-23,crx-12,crx-01,ry,crx-32,crx-03,crx-10,crx-21",
    "ry,cnot-30,cnot-23,cnot-12,cnot-01,ry,cnot-32,cnot-03,cnot-10,cnot-21",
    "rx,rz,crz-10,crz-32,crz-21",
    "rx,rz,crx-10,crx-32,crx-21",
    "rx,rz,crz-30,crz-23,crz-12,crz-01",
    "rx,rz,crx-30,crx-23,crx-12,crx-01",
]
