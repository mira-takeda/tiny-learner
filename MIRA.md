# MIRA.md — Handoff para trabajar tiny-learner en Windows

> Lee este archivo primero. Es la guía para que Muse/Mira entienda
> TODO el proyecto sin leer los 41 MB de logs. Fuente de verdad
> operativa: `CROSSCHECK.md`. Este archivo es el mapa + setup Windows.
> Creado 2026-10-30 en el Mac; proyecto copiado a Windows por red local.

## 1. Qué es esto (30 segundos)

- **tiny-learner**: un experimento pequeño y falsable para el
  Meta Global AI Developer Hackathon (26 oct – 4 nov 2026, quedan ~4 días).
- Hipótesis: una red modular diminuta (**7,944 params, NO transformer**)
  aprende skills atómicas (A, B, C, D) solo con reward escalar y luego
  resuelve combinaciones NUNCA VISTAS (H1–H3) mejor que un monolito
  de igual tamaño.
- Todo corre en CPU, determinista, sin llamadas a APIs durante episodios.
- Claim honesto actual: selección aprendida útil (A01), NO composición
  general ni transferencia. Sin claims de emergencia/AGI.

## 2. Estado actual (no re-derivar, está cerrado)

- **S1–S16 cerrados, A01 congelado.** Números: pilot3 A=1.00/B=0.83/
  C=0.843/D=1.00; v2 I1=0.047 (ininterpretable); v2conf ACQ=0.102 FAIL;
  **A01 = 0.647 en ACQ18 vs A-only 0.145 / B-only 0.246.**
- Campaña registrada: H-main INCONCLUSIVE (C no adquirido 20/20),
  H-iso FAIL, H-meta INCONCLUSIVE. Detalle en `HANDOFF.md` + `RUNLOG.md`.
- **Pendientes**: monolito corregido; paso W1 (necesita GO explícito
  del usuario); demo de 3 min con claim honesto.

## 3. Reglas vinculantes (no romper)

1. **Fuente de verdad = `CROSSCHECK.md`.** `HANDOFF.md` es historial.
   `PREREG.md` congelado (no tocar). `RUNLOG.md` solo-append.
2. **Regla 4 días**: NINGUNA cadena nueva de sondas, NINGÚN S17,
   NINGÚN tuning, hasta que el usuario elija: (A) una última prueba
   pre-congelada o (B) cerrar y pasar a demo/submission.
3. **Código congelado**: `src/` congelado en commit `e8f3137`. No editar
   `src/`, `data/*.json` sin una decisión registrada. Docs/scripts
   nuevos a nivel raíz sí están permitidos.
4. **Separación de prior-work**: TAVI/FOCO/Semilla son previos, NO son
   hackathon. No copiar su código aquí. Todo vive en `tiny-learner/`.
5. **Aprendizaje en el sistema pequeño**: nunca simular resultados con
   respuestas del asistente. Todo número sale de ejecución real local.
6. **Sin secretos en repo**: credenciales Modal (`~/.modal.toml`) y
   tokens quedan FUERA. La repro no necesita Modal ni red.

## 4. Mapa de archivos (qué es cada cosa)

```
tiny-learner/
├── MIRA.md              ← ESTE archivo (empieza aquí en Windows)
├── CROSSCHECK.md        ← estado operativo actual (fuente de verdad)
├── HANDOFF.md           ← historial técnico por sonda (S1–S16, N0–N5)
├── CONTEXT.md           ← contexto vivo del proyecto (diseño, constraints)
├── PREREG.md            ← pre-registro CONGELADO 2026-10-26 (no tocar)
├── RUNLOG.md            ← bitácora append-only de todo lo ejecutado
├── README.md            ← estado + repro mínima
├── requirements.txt     ← deps raíz (solo numpy, = repro)
├── setup_windows.ps1 / setup_windows.bat  ← setup+verificación en 1 paso
├── repro_s16/           ← REPRO MÍNIMA (A01 vs A/B, 3600 eps, ~60 s)
│   ├── repro.py, requirements.txt, MANIFEST.json (10 hashes), README.md
│   └── PROVENANCE.md (roles de Muse, evidencia para entrega)
├── src/                 ← sistema congelado: env, models, tasks, gen,
│   │                      evaluate, audit, es, train, selftest (NO TOCAR)
├── data/                ← datasets congelados + MANIFEST (no tocar)
│   ├── test_battery_v1.json, train_meta/control_v1.json
├── pilot3/, v2/, v2conf/, v2views/   ← código de campañas follow-up
├── *_runs/, diag_runs/, runs_modal/  ← artefactos .npz/.json (pesados,
│   │   ignorados por git; necesarios para re-análisis, NO para repro)
├── v2conf_diag/         ← diagnósticos S1–S16 (~39 MB, con freezes sellados)
├── diag/ + pilot/, pilot2/, c5/, c6/ ← líneas previas / auxiliares
├── modal_*.py           ← wrappers nube usados 2026-10-27 (histórico;
│                           la repro local NO los necesita)
└── .git/                ← historia hasta campaña registrada; el trabajo
                            S1–S16 vive en archivos untracked (ver nota §7)
```

## 5. Setup en Windows (paso a paso)

Requisitos: **Python 3.12+** (ideal 3.13/3.14; numpy 2.5.4 NO instala
en 3.11 — verificado en este proyecto), ~200 MB libres, sin GPU.

Opción rápida (desde `tiny-learner/` en PowerShell):

```powershell
.\setup_windows.ps1
```

O manual:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r repro_s16\requirements.txt
.venv\Scripts\python repro_s16\repro.py
```

Verificación: debe imprimir `hashes ok: 10 files`, la tabla por theta
y `A01 0.6467 / A-ONLY 0.1450 / B-ONLY 0.2458`, ~60–120 s según CPU.
Si los hashes fallan → la copia está corrupta, re-copiar el zip.

Diferencias Windows vs Mac (importante para Mira):
- Mac usaba `python3.14` explícito; en Windows es `python` (verificar
  `python --version` ≥ 3.12, porque `python3` a veces no existe).
- Activar venv: `.venv\Scripts\Activate.ps1` (no `bin/activate`).
- Rutas con `\`; el código Python usa rutas relativas, funciona igual.
- `modal` CLI no instalado allÁ (no importa: solo se usa histórico).
- Git: configurar identidad antes de commitear
  (`git config user.name/email`); NO hay remoto GitHub todavía.

## 6. Qué hacer allá (siguiente paso autorizado)

1. Verificar repro (§5) → confirmar entorno sano.
2. NO diseñar experimentos hasta que el usuario elija (A) o (B) (§3.2).
3. Si elige (A): W1 = ensanchamiento procedural de G1 (d_gh ~10–35,
   ángulos pequeños, MISMA física/rewards, ciego a IDs de I1),
   re-entrenar coordinador Tiny original, gate en batería FRESCA +
   H1/H2/H3 una sola vez (I1 retirado como gate). Generador + barra
   se congelan ANTES de correr. Necesita GO explícito.
4. Si elige (B): demo 3 min (bet → antes/después → A01 money shot →
   barras honestas → claim + 1 control) + empaquetar submission.

## 7. Notas de transferencia (Mac → Windows)

- Este paquete es COPIA de archivos (zip), no clon git: el trabajo
  S1–S16 está untracked en el Mac, así que `git archive`/push perdería
  casi todo. El `.git/` incluido conserva la historia congelada.
- Excluido del zip: `__pycache__/`, `.venv/`, `.DS_Store`.
- La repro (`repro_s16/`) es autocontenida: solo necesita Python +
  numpy. El resto de carpetas `*_runs`/`v2conf_diag` solo hacen falta
  para re-análisis o continuación científica (W1).
- Si algo no cuadra entre máquinas, el árbitro es `repro_s16/repro.py`:
  misma salida = mismo proyecto.
