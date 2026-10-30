<div align="center">

<sub>RESEARCH / ADAPTIVE SYSTEMS / 01</sub>

# TINY LEARNER

### Learn. Remember. Adapt.

**A compact learning system that explores how new abilities can be learned,<br>
retained, and reused as conditions change.**

<sub>CPU-FIRST &nbsp; / &nbsp; CONTINUAL LEARNING &nbsp; / &nbsp; SKILL TRANSFER</sub>

<sub>META GLOBAL AI DEVELOPER HACKATHON 2026</sub>

[Explore the learning process](#explore-the-learning-process) &nbsp; · &nbsp; [Applications](#where-it-could-matter) &nbsp; · &nbsp; [If it scales](#if-it-scales)

</div>

---

## What this could become

**A growing repertoire of skills. A compact learner that keeps adapting.**

The interactive concept explores three possible directions for Tiny Learner: adaptive robotic manipulation, autonomous inspection, and cooperation among specialized learners.

[Explore the future of Tiny Learner](https://mira-takeda.github.io/tiny-learner/)  
<sub>The interactive site becomes accessible once GitHub Pages is enabled for this repository.</sub>

## Beyond fixed instructions

A conventional automation can follow an impressive set of instructions. But when its environment changes, those instructions may no longer cover the situation.

Tiny Learner investigates a different possibility: **a small system that keeps learning, carries useful skills forward, and applies past experience to challenges it has not practiced in exactly the same form.**

<table>
<tr>
<td align="center" width="33%"><strong>LEARN</strong><br><sub>Acquire behavior through experience</sub></td>
<td align="center" width="33%"><strong>REMEMBER</strong><br><sub>Keep earlier abilities while learning more</sub></td>
<td align="center" width="33%"><strong>ADAPT</strong><br><sub>Apply experience under new conditions</sub></td>
</tr>
</table>

## Explore the learning process

**Open any stage to see what the system is being asked to do.** The stages describe the experimental challenge, not a fixed sequence of scripted answers.

<details open>
<summary><strong>01 / EXPERIENCE</strong> · Acquiring a behavior</summary>

A learner encounters a task, takes actions, and receives feedback. Its internal state can change with experience.

**Example:** a robotic manipulator practices how to grasp objects with different shapes and orientations.

The key question is whether useful behavior is learned from experience rather than supplied as a separate rule for every case.

</details>

<details>
<summary><strong>02 / CONTINUITY</strong> · Learning after learning</summary>

New tasks arrive after earlier training. The same learner continues updating.

**Example:** after practicing grasping, the system encounters a sorting task with changing destinations and constraints.

The challenge is not simply acquiring the new behavior. It is doing so **without needlessly disrupting what was learned earlier**.

</details>

<details>
<summary><strong>03 / TRANSFER</strong> · Encountering unfamiliar conditions</summary>

The environment changes, and the learner faces situations that were not directly rehearsed as complete solutions.

**Example:** the objects, target stations, and ordering of operations change together.

Can previously acquired abilities still contribute to the new task, without training a new solution for each combination?

</details>

<details>
<summary><strong>04 / EVALUATION</strong> · Finding out what really carried over</summary>

After continued learning, evaluation addresses different questions separately:

- **Acquisition:** what new behavior was learned?
- **Retention:** which earlier abilities remain?
- **Transfer:** what works in unseen situations?
- **Interference:** what changed or degraded after further learning?

Controlled tasks and held-out tests help distinguish genuine reuse from performance on already practiced cases.

</details>

## Where it could matter

The potential applications share a common need: **machines that can reuse learned behavior when their tasks or surroundings change.**

<details>
<summary><strong>Robotic manipulation</strong> · Handling changing objects and workflows</summary>

A robotic arm could accumulate useful grasping, sorting, and placement behaviors, then draw on them when new parts or sequences are introduced.

</details>

<details>
<summary><strong>Inspection and maintenance</strong> · Working beyond a fixed route</summary>

A mobile inspection system could combine movement, measurement, and task-specific responses across changing layouts and equipment.

</details>

<details>
<summary><strong>Agricultural robotics</strong> · Adapting to changing field conditions</summary>

Compact field machines could build upon learned sensing, movement, and handling behaviors as crops, terrain, and operating conditions vary.

</details>

<details>
<summary><strong>Warehouse and workshop helpers</strong> · New tasks from familiar abilities</summary>

A helper robot could reuse pickup, transport, and placement skills when destinations, obstacles, and task priorities change.

</details>

<details>
<summary><strong>Embedded and service robotics</strong> · Learning close to the device</summary>

Smaller machines, including kitchen and service assistants, could adapt recurring behaviors locally without calling a large remote model for every routine decision.

</details>

## If it scales

**More skills. Longer experience. Richer environments.**

The longer-term question is whether a compact learning mechanism can support a growing repertoire of abilities, more complex situations, and cooperation among specialized learners.

Scaling might mean more than increasing the number of parameters. It could also mean **making better use of experience** as the range of tasks expands.

---

## Built for ordinary hardware

Tiny Learner follows a **CPU-first, small-footprint** approach. This keeps experiments accessible on ordinary computers and opens up possibilities for resource-constrained devices.

## Muse and the curriculum

Meta Muse contributes to the development of learning challenges and experimental curricula. Tiny Learner is the system whose skill acquisition, retention, and transfer are studied.

## Research notes

**Active experimental prototype.** The work focuses on learning over time, interference, retention, and generalization.

<details>
<summary><strong>Technical evaluation methodology</strong></summary>

Experiments use defined controls, separate checks for retention and transfer, and held-out tasks. When applicable, protocols distinguish `PASS`, `FAIL`, and `UNKNOWN`. Specific results belong with their experiment configurations and evaluation criteria.

</details>

---

<div align="center">

<sub>LEARNING / MEMORY / ADAPTATION</sub>

[Research profile](https://github.com/mira-takeda) &nbsp; · &nbsp; [Back to top](#tiny-learner)

</div>

## Experimental code and reproducibility

- [Local research README: results, reproduction instructions and documentation map](README.local.md)
- [Current operational state](CROSSCHECK.md), [research context](CONTEXT.md), and [technical handoff](HANDOFF.md)
- [Frozen preregistration](PREREG.md) and [append-only run history](RUNLOG.md)
- [Reproduction package and provenance](repro_s16/README.md)
- [Frozen source and checks](src/) and [datasets](data/)

The research README records measured results and their limitations. The site presents the research direction and possible future applications. Existing experimental code, tests, and saved artifacts are retained for provenance; integration runs no new experiments.