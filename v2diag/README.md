# v2diag/ — registro de diagnósticos post-v2 (medición, sin entreno)

Artefactos frozen (v2_runs/, pilot3_runs/, datos) intactos: todo aquí
es sólo-lectura sobre ellos. Scripts y registros de cada diagnóstico.

## D1. Trazas del gate (trace_gate.py, records_traces.json)

27 episodios fijos pre-run: baterías G1H/I1/H1 × primeras 3 entradas
(H-G1-000{0,1,2}, T-I1-000{0,1,2}, T-H1-000{0,1,2}) × thetas V2
301–303; misma política sampled y semillas del scoring.
Registra por paso: dists obs, pesos gate, argmax/sharpness por skill,
argmax mezcla, acción; causa terminal.
Resultado: G1H→I1 sin cambio de routing (A/D-dominante, w_B≈0);
éxitos G1 con w_B≈0; H1 recluta C (0.39) y B débil cerca del hazard.
Fórmula mezcla verificada p=Σw·softmax; con w_B≥0.3 el blend sigue
a B ~90%. Trazas apoyan hipótesis, no prueban causalidad.

## D2. Teleport geométrico (teleport.py)

Mismos 9 I1. Reproducción 9/9 exacta. Regla determinista fijada
pre-resultados: rotar hazard sobre el agente (conserva |A-H| exacto),
primer ángulo en [180,90,−90,135,−135,45,−45] con beacon–haz≥35 y
haz en [5,95]²; velocidad/agente/beacon/semilla/política/horizonte
idénticos. Aplicados: −90°/+45°/−90° (db 21→71, 20→36, 14→85).
Resultado: 0/9 → 7/9 llegadas; w_B≈0 también en recuperadas.
PRECISIÓN: la intervención cambia posición angular + relación con el
beacon conservando distancia al agente; NO identifica exclusivamente
el efecto de una distancia escalar. Apoya sensibilidad causal a esa
geometría en esos episodios; no demuestra coordinación con B, no
resuelve H1, no cambia el veredicto v2 (INCONCLUSIVE).

## D3. Sonda B one-hot (b_probe.py)

Mismos 9 I1 ORIGINALES (sin teleport), skill-1 pura sampled, mismos
thetas/semillas/horizonte. Sin switching.
Resultado: 7 colisiones, 1 llegada (0002/301, T65), 1 supervivencia
a horizonte (0002/303, T200). B dura 3–4× más que el gate (T 22–32
vs 7–14) pero colisiona en 7/9.
Lectura preregistrada: B NO evita de forma fiable estas colisiones;
pendiente distinguir generalización de B vs dificultad/factibilidad;
NO concluir imposibilidad. Hipótesis abierta anotada: B se entrenó
con slot S1≡0 (sin beacon) y en I1 S1 está activo (input OOD para
la propia skill), además de la geometría de guarda.

## D4. Sonda máscara-S1 en B (mask_s1_probe.py)

Premisa verificada pre-run: 400/400 cfgs B sin 'goal' ⇒ S1=obs[2:5]
=[0,0,0] exactos en todo el entreno B. Mismos 9 I1, B one-hot,
intervención única obs[2:5]=0 ante B (mundo intacto); resto idéntico.
UNMASK reproduce D3 exacto. MASK: colisiones 7/9→1/9 (7 supervivencias
T200 + 1 llegada 0002/302; 0000/301 sigue colisionando T20). TV
intra-paso grande donde importaba (0.57–0.79, acuerdo-argmax 0.15–0.36);
pequeña donde B ya sobrevivía. 0002/301: llegada→supervivencia (llegadas
bajo B son incidentales en ambos sentidos).
Lectura preregistrada: apoya interferencia causal de S1 sobre B en
estos escenarios; 0000/301 muestra que no basta en todos (otras
diferencias pendientes); no prueba imposibilidad ni composición.

## D5. Confirmatoria máscara-S1 (i1_confirm.py)

Máscara fija D4, resto 197 I1 (índices 3..199) × thetas 301–303,
B one-hot, sin selección. Exploratorias aparte (no agregadas).
Colisiones 197: 301: 188→36; 302: 166→17; 303: 148→21 (métrica
primaria). Flips pareados col→no-col: 153/149/128; reversos: 1/0/1.
Duración media +92/+105/+86 pasos (registro crudo: +91.9/+104.7/+85.5;
corregido: decía +86 para 301). Llegadas/supervivencias suben
(llegada no primaria bajo B). Conclusión acotada D4 confirmada.

## D6. Auditoría contrato obs (inline) + vistas (VIEWS.md)

A: goal+vel; B: hazard+vel; C: obj+zone+vel; D: goal+vel (400/400
cada uno); ausencia=zeros(3) exactos. Propuesta: vistas por skill =
slots de su entreno + vel; gate completo. Respaldada: B–S1. Resto
sin probar. No aplicada; nada congelado tocado; sin reentreno.
