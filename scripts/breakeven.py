#!/usr/bin/env python3
"""Punto de equilibrio económico de Redukt: coste de cómputo vs ahorro de almacenamiento.

Responde a la pregunta: ¿a qué throughput (bytes/s por vCPU) comprimir 1 GB cuesta menos que el
almacenamiento que ese GB ahorra durante T años? Contexto en
docs/internal/research/throughput-properties-audit.md (sección 4) y ROADMAP Fase 19.

Modelo (por GB de entrada, raw):
    C_comp = P_cpu * 1e9 / (3600 * R)                       coste de cómputo; R en bytes/s por vCPU
    A_stor = S * b * P_st * 12 * T                          ahorro de almacenamiento
    R*     = P_cpu * 1e9 / (3600 * S * b * P_st * 12 * T)   throughput mínimo de equilibrio
donde P_cpu = $/vCPU-hora, P_st = $/GB-mes, T = años, b = fracción que deja el baseline (bpb de brotli / 8)
y S = ahorro adicional de Redukt frente a ese baseline.

Uso:
    python scripts/breakeven.py

Sin argumentos ni dependencias (solo stdlib). Imprime a stdout:
    1. coste $/GB raw por throughput y tipo de instancia;
    2. ahorro de almacenamiento por clase de almacenamiento, S y T;
    3. R* (KB/s por vCPU) con spot $0.022 y con on-demand $0.036;
    4. caso Redukt (enwik8, apache, facturae, dickens) con b y S medidos.

Para otros escenarios, editar los diccionarios `cpu` y `stor` y la lista del caso Redukt al final del script.
Precios de referencia: 2026-11 (instances.vantage.sh); Hetzner está en EUR y se trata como USD (aproximación).
"""
# Break-even: compute cost to compress 1 GB (raw) vs storage saving over T years.
# C_comp = P_cpu * 1e9 / (3600 * R)         [$ per raw GB; R bytes/s per vCPU]
# A_stor = S * b * P_st * 12 * T             [$ per raw GB; b = baseline brotli fraction, S = saving vs baseline]
# R*     = P_cpu * 1e9 / (3600 * S * b * P_st * 12 * T)
GB=1e9
cpu={'spot(c7g/c7a/m7i ~$0.022)':0.022,'on-demand c7g $0.036':0.03625,'on-demand m7i $0.050':0.0504,'dedicated Hetzner AX42 ~EUR0.0068/thread-h':0.0068}
stor={'S3 Standard $0.023':0.023,'S3 Deep Archive $0.00099':0.00099,'On-prem $50/TB/yr':50/1000/12}
rates=[100,1e3,1e4,1e5,1e6]
print("Compute cost $/GB raw by throughput:")
for r in rates:
    hrs=GB/r/3600
    print(f"  R={r/1e3:>7.1f} KB/s  hours/GB={hrs:>8.1f}  "+"  ".join(f"{k.split()[0]}=${v*hrs:,.3f}" for k,v in cpu.items()))
print("\nRedukt today 130 B/s:", f"hours/GB={GB/130/3600:.0f}", {k:round(v*GB/130/3600,1) for k,v in cpu.items()})
print("\nStorage saving $/GB (b=1) by class, S, T:")
for sk,sp in stor.items():
    for S in (0.3,0.5,0.8):
        print(f"  {sk:28s} S={S:.0%} "+"  ".join(f"{T}y=${S*sp*12*T:.4f}" for T in (1,3,7)))
print("\nBreak-even R* (KB/s per vCPU, b=1, spot $0.022):")
P=0.022
for sk,sp in stor.items():
    for S in (0.3,0.5,0.8):
        print(f"  {sk:28s} S={S:.0%} "+"  ".join(f"{T}y={P*GB/(3600*S*sp*12*T)/1e3:>8.1f}" for T in (1,3,7)))
print("\nBreak-even R* (KB/s per vCPU, b=1, on-demand $0.036):")
P=0.03625
for sk,sp in stor.items():
    for S in (0.3,0.5,0.8):
        print(f"  {sk:28s} S={S:.0%} "+"  ".join(f"{T}y={P*GB/(3600*S*sp*12*T)/1e3:>8.1f}" for T in (1,3,7)))
print("\nEgress: saving per download per GB raw = S*b*0.09; months of S3 Std equivalent = 0.09/0.023 =", round(0.09/0.023,2))
print("\nRedukt-specific (b from brotli bpb/8):")
for name,bpb_br,S in (('enwik8 text',2.352,0.59),('apache log',0.466,0.54),('facturae xml',0.890,0.56),('dickens',2.778,0.77)):
    b=bpb_br/8
    for sk,sp in stor.items():
        A7=S*b*sp*12*7
        Rstar=0.022*GB/(3600*A7)
        print(f"  {name:13s} b={b:.3f} S={S:.0%} {sk:26s} save7y=${A7:.4f}/GB raw  R*(7y,spot)={Rstar/1e3:.1f} KB/s  x{Rstar/130:.0f} vs 130 B/s")
