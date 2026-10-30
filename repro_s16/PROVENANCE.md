# Procedencia: curriculo/diseno con Muse + evidencias

## Roles (registrado en repo)

- CONTEXT.md S2: el aprendizaje ocurre en el sistema pequeno;
  nunca simulado por respuestas de Muse. Meta/Muse permitidos
  como maestro, generador de experiencias, evaluador, tester
  adversarial y especialista.
- PREREG.md S4: curriculo Meta generado por Meta Muse Spark
  (`muse-spark-1.3-contributor`) solo con esquema + defs de
  skills, sin ver tests; snapshot antes de muestrear tests;
  auditoria de leakage automatica. Limitacion honesta
  declarada: generador e implementador comparten pesos.
- CONTEXT.md S4: entorno auditado (Muse Code 1.4.13, provider
  meta, mismo modelo; Intel Mac CPU; python3.14 + numpy 2.5.4).
- RUNLOG.md: snapshot de curriculo con hashes (2026-10-26),
  entrenamiento registrado 20/20 (2026-10-27), veredictos
  H-main/H-meta INCONCLUSIVE, H-iso FAIL.

## Por etapa (no extender ciegos fuera de su etapa)

1. Curriculo original: generado por Meta Muse Spark con
   esquema+skills, ciego a tests (PREREG.md S4, snapshot con
   hashes en RUNLOG.md 2026-10-26). La declaracion de tests
   ciegos cubre SOLO este curriculo original.
2. Diagnosticos S1-S16: protocolos propuestos/disenados con
   asistencia de Muse Code (mismo modelo); cada GO, tope y
   congelado consta en v2conf_diag (freeze*.json, README.md)
   y HANDOFF.md. Aqui el asistente participo en el diseno
   iterativo con resultados a la vista -- sin declaracion de
   ceguera; la validez descansa en sellados pre-registro
   (datos/congelados antes de ejecutar) y baterias frescas.
3. Candidato final A01: pesos S14 (alpha .01) + confirmacion
   S15 en ACQ18 nuevo y S16 vs A/B; repro local determinista
   (esta carpeta). Pesos, escenarios y hashes en repo.
- Toda ejecucion numerica corrio localmente en CPU con codigo
  congelado; cero llamadas por episodio.
- Garantia de test-time (CONTEXT.md S5): la evaluacion solo
  invoca al sistema pequeno.
- Transcripciones de sesion: disponibles solo como logs
  locales de la herramienta (no en repo, sin secretos que
  conservar); el registro durable es RUNLOG/HANDOFF/freezes.

## Rutas con clave (revisado 2026-10-30)

- Existe `~/.modal.toml` (109 bytes, 2026-10-26; credenciales
  Modal fuera del repo, usadas para el entrenamiento registrado
  2026-10-27; contenido jamas leido). Formato verificado contra
  el cliente Modal publico instalado: seccion de perfil con
  `token_id` + `token_secret`. Plantilla sin secretos:
  repro_s16/modal.toml.template (recrear con `modal token new`).
  La repro NO la necesita ni la lee.
- Sin variables KEY/TOKEN en entorno; sin SDKs/MCP de Meta
  configurados (CONTEXT.md S4, vigente).
- Git tiene identidad local (Satoru Takeda, noreply); `gh`
  ausente. Subida a GitHub pendiente de entrega (falta repo
  remoto + video demo); sin instrucciones de integracion no
  probada.

## Evidencia que falta para la entrega

- Re-ejecucion independiente en una segunda maquina (la repro
  se corrio aqui: repo 58 s + copia limpia 60.1 s).
- Lockfile de dependencias (constan versiones numpy==2.5.4,
  python 3.14.6; sin pin completo de entorno).
- Curvas por epoca S8/S10/S11: limitacion documental
  historica, sin reconstruir. Estados visitados ACQ14
  tampoco se guardaron.
- Transcripciones de sesion fuera de RUNLOG (logs locales de
  la herramienta, no parte del repo).
- Repro de entrenamiento S13/S14 (scripts en repo con topes
  900/300 s; la demo actual reproduce solo evaluacion).
