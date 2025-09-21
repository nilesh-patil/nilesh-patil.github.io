---
layout: single
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume/
  - /resume
toc: true
toc_sticky: true
---

**Download** - [PDF version of this CV]({{ '/files/nilesh-patil.pdf' | relative_url }})
{: .notice--info}

> The web version above is authoritative; the PDF may lag the web by a few weeks.

## Research interests

Primary focus is on AI/ML for Science, with a strong inclination towards computational biology. I study & experiment on how to make learning systems that do science reliably : plan and run experiments over long horizons, model complex systems, and report how certain they are.

- **Agents for scientific discovery.** Long-horizon planning and hypothesis-driven experimentation by LLM agents, with traceable provenance, verifiable evidence, and specialized critics that catch confounded lines of inquiry.
- **Generative models and simulators of complex systems.** Learned simulators of cellular and tissue-level behaviour (perturbation response, cell-state trajectories, spatial and imaging data such as multiplexed and hyperspectral microscopy) that serve as testbeds where hypotheses and agents can be evaluated in silico.
- **Reliability under shift.** Calibration and uncertainty, evaluation in nonstationary environments, and domain adaptation of small open-weight models for specialised scientific domains.
- **Foundations I bring from industry :** sequential decision-making (deep RL, contextual and non-stationary bandits), population-scale behavioural simulation, post-training large models and large-scale ML systems. Open to adjacent quantitative domains ( chemistry, materials, neuroscience, climate etc ) where the same opportunities exist in scientific simulations.

## Education
<div class="cv-entry">
<div class="cv-entry__when">2017</div>
<div class="cv-entry__what" markdown="1">

**M.S. in Data Science** - University of Rochester &nbsp;·&nbsp; *Rochester, New York*

</div>
</div>
<div class="cv-entry">
<div class="cv-entry__when">2013</div>
<div class="cv-entry__what" markdown="1">

**Bachelor of Technology in Metallurgical & Materials Engineering** - Indian Institute of Technology &nbsp;·&nbsp; *Roorkee, India*

</div>
</div>

## Research experience
<div class="cv-entry">
<div class="cv-entry__when">2024 - 2026</div>
<div class="cv-entry__what" markdown="1">

**Senior Principal Research Scientist** - Dream Sports &nbsp;·&nbsp; *New York / Mumbai*

- Established and led an industry-academia research partnership with **Columbia University**, co-defining research agendas at the intersection of ML and sports analytics and supporting joint work with faculty, postdocs, and PhD students.
- Defined a multi-year applied-research portfolio - formulating ~23 candidate research problems and securing funding for 10 - spanning sports robotics, LLM-based persona/behavior simulators, and agentic evaluators for personalization.
- Developed deep-learning models for early churn prediction from large-scale user-product interaction time series (published, ICMLA 2023), and LLM-based behavioral simulation of user lifetime trajectories at population scale (250M+ users).
- Built distributed recommendation, semantic similarity search over ~100M entities, and feature infrastructure for 250M+ users - work that seeded publications on personalization (UMAP '25, ECML-PKDD '25) and entity resolution (ACL 2026).
- Designed a real-time probabilistic forecasting system producing 50k+ concurrent forecasts for high-performance production ML applications.

</div>
</div>
<div class="cv-entry">
<div class="cv-entry__when">2017 - 2019</div>
<div class="cv-entry__what" markdown="1">

**Staff Data Scientist** - Center for Vaccine Biology & Immunology &nbsp;·&nbsp; *Rochester, NY*

- Developed deep-learning pipelines for biomedical image analysis - including 3D reconstruction from hyperspectral microscopy, contributing computational methods to immunology research published in *Cell Reports* and the *Journal of Immunology*.
- Automated ultrasound Doppler-angle estimation using DNNs on clinical ultrasound scans (EMBC 2019).

</div>
</div>
<div class="cv-entry">
<div class="cv-entry__when">Dec 2025 - Apr 2026</div>
<div class="cv-entry__what" markdown="1">

**Independent projects**

- **Extending SimuCell3D** [ [Runser et al. (2024)](https://www.nature.com/articles/s43588-024-00620-9) ] - a high-performance C++ framework for simulating 3D tissue mechanics at subcellular resolution. My performance-focused [fork](https://github.com/nilesh-patil/simucell3d) extends the published version with OpenMP scheduling and contact-detection changes. [Write-up]({{ '/posts/simucell3d-adaptive-openmp-scheduling/' | relative_url }}).

</div>
</div>

## Employment history
<div class="cv-entry">
<div class="cv-entry__when">2026 - Present</div>
<div class="cv-entry__what" markdown="1">

**Head of AI** - DreamStreet &nbsp;·&nbsp; *Mumbai*

- **Long-horizon planning benchmark :** designed a verifiable, judge-free benchmark for long-horizon planning by LLM agents in nonstationary environments, using sequential decisions over hundreds of steps with delayed rewards and tools whose reliability drifts in hidden ways. Characterised how frontier models commit to, revise, and abandon strategies across regime changes, over 4 models × 8 rollouts.
- **Research-native agent harness.** Built a research-native agent harness for autonomous, hypothesis-driven experimentation. It has parallel forkable workers in isolated sandboxes, an append-only graph linking hypotheses, experiments, and artifacts for full provenance, sealed execution receipts, and a memory-free critic that breaks the agent out of local search loops.
- Designed a compliance-aware orchestration framework for multi-step LLM agents in a SEBI-regulated financial setting. It combines constrained tool use, policy enforcement, and end-to-end audit logging, which makes agent decisions controllable and auditable.
- Built self-hosted LLM infrastructure for in-house, post-trained open-weight LLMs (Qwen 3.6 27B / Qwen3.6-35B-A3B on vLLM, across GCP and AWS). It is a reproducible substrate for agent experiments, domain adaptation, and self-serve research agents ( quantitative trading, literature and domain reviews, world event monitoring ).

</div>
</div>
<div class="cv-entry">
<div class="cv-entry__when">2019 - 2026</div>
<div class="cv-entry__what" markdown="1">

**Senior Principal Research Scientist** - Dream11 &nbsp;·&nbsp; *Mumbai*

- Led applied AI research for Dream Sports and built a high-performing cross-continent team of research scientists, applied scientists, and ML engineers across India and New York.
- Delivered regular data, ML, and AI training sessions across Dream Sports for audiences ranging from ~10 to 200 participants.
- Led the team working on distributed forecasting & recommendation systems, text similarity search for ~100M entities, and feature-store systems supporting 250M+ users.

</div>
</div>
<div class="cv-entry">
<div class="cv-entry__when">2017 - 2019</div>
<div class="cv-entry__what" markdown="1">

**Staff Data Scientist** - David H. Smith Center for Vaccine Biology and Immunology, University of Rochester &nbsp;·&nbsp; *Rochester, NY*

- Built automated and self-serve ML systems for bio-imaging research, including 3D reconstruction from hyper-spectral microscopic scans.

</div>
</div>
<div class="cv-entry">
<div class="cv-entry__when">2014 - 2016</div>
<div class="cv-entry__what" markdown="1">

**Data Scientist** - AXA Insurance &nbsp;·&nbsp; *Pune, India*

- Built survival / mortality-forecasting models and scaled statistical pipelines (Spark, Python) for US population ( ~300M+ entities ).

</div>
</div>
<div class="cv-entry">
<div class="cv-entry__when">2013 - 2014</div>
<div class="cv-entry__what" markdown="1">

**Data Analyst** - AbsolutData Research & Analytics &nbsp;·&nbsp; *Gurgaon, India*

- Developed a multi-stage predictive-maintenance model fusing sensor telemetry, oil-analysis, and human-labeled alerts.

</div>
</div>

## Selected publications

For the full, continuously-updated list, see [Publications]({{ '/publications/' | relative_url }}) or [Google Scholar](https://scholar.google.co.in/citations?user=IIabY1sAAAAJ).

<div class="cv-entry">
<div class="cv-entry__when">2026</div>
<div class="cv-entry__what" markdown="1">

**Structure-Guided Entity Resolution: Fine-Tuning LLMs for Robust Name Matching in Complex Linguistic Contexts.** Chourasia, A., Kapoor, S., &amp; Patil, N. - Association for Computational Linguistics (ACL), Industry Track. [doi.org/10.18653/v1/2026.acl-industry.101](https://doi.org/10.18653/v1/2026.acl-industry.101)

</div>
</div>
<div class="cv-entry">
<div class="cv-entry__when">2024</div>
<div class="cv-entry__what" markdown="1">

**Optimizing Fantasy Sports Team Selection with Deep Reinforcement Learning.** Proceedings of CODS-COMAD '24, ACM, 284-291. [doi.org/10.1145/3703323.3703743](https://doi.org/10.1145/3703323.3703743)

</div>
</div>
<div class="cv-entry">
<div class="cv-entry__when">2023</div>
<div class="cv-entry__what" markdown="1">

**Early Churn Prediction from Large Scale User-Product Interaction Time Series.** 2023 International Conference on Machine Learning and Applications (ICMLA), IEEE. [doi.org/10.1109/ICMLA58977.2023.00314](https://doi.org/10.1109/ICMLA58977.2023.00314)

</div>
</div>
<div class="cv-entry">
<div class="cv-entry__when">2021</div>
<div class="cv-entry__what" markdown="1">

**CXCL10+ Peripheral Activation Niches Couple Preferred Sites of Th1 Entry with Optimal APC Encounter.** Cell Reports. [doi.org/10.1016/j.celrep.2021.109523](https://doi.org/10.1016/j.celrep.2021.109523)

</div>
</div>
<div class="cv-entry">
<div class="cv-entry__when">2020</div>
<div class="cv-entry__what" markdown="1">

**CXCL10+ Perivascular Clusters Nucleate Th1 Cell Tissue Entry and Activation in the Inflamed Skin.** Journal of Immunology. [doi.org/10.4049/jimmunol.204.Supp.220.9](https://doi.org/10.4049/jimmunol.204.Supp.220.9)

</div>
</div>
<div class="cv-entry">
<div class="cv-entry__when">2019</div>
<div class="cv-entry__what" markdown="1">

**Automated Ultrasound Doppler Angle Estimation Using Deep Learning.** 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), pp. 28-31, IEEE. [pubmed.ncbi.nlm.nih.gov/31945837](https://pubmed.ncbi.nlm.nih.gov/31945837)

</div>
</div>

## Research supervision &amp; mentorship

Publications led with researchers and engineers I mentored at Dream Sports:

<div class="cv-entry">
<div class="cv-entry__when">2026</div>
<div class="cv-entry__what" markdown="1">

**Driving Engagement in Daily Fantasy Sports with a Scalable and Urgency-Aware Ranking Engine.** Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, No. 47. [doi.org/10.1609/aaai.v40i47.41479](https://doi.org/10.1609/aaai.v40i47.41479)

</div>
</div>

<div class="cv-entry">
<div class="cv-entry__when">2025</div>
<div class="cv-entry__what" markdown="1">

**Enhancing Personalisation in Fantasy Sports with Graph-Based Representations.** UMAP '25, ACM, 361-363. [doi.org/10.1145/3699682.3730974](https://doi.org/10.1145/3699682.3730974)

</div>
</div>

<div class="cv-entry">
<div class="cv-entry__when">2025</div>
<div class="cv-entry__what" markdown="1">

**Personalized Contest Recommendation in Fantasy Sports.** ECML PKDD 2025, Applied Data Science Track, Part IX, Springer, 22-35. [doi.org/10.1007/978-3-032-06118-8_2](https://doi.org/10.1007/978-3-032-06118-8_2)

</div>
</div>

<div class="cv-entry">
<div class="cv-entry__when">2024</div>
<div class="cv-entry__what" markdown="1">

**ForeCal: Random Forest-based Calibration for DNNs.** CODS-COMAD '24, ACM, 44-51. [doi.org/10.1145/3703323.3703330](https://doi.org/10.1145/3703323.3703330)

</div>
</div>

<div class="cv-entry">
<div class="cv-entry__when">2023</div>
<div class="cv-entry__what" markdown="1">

**FENCE: Fairplay Ensuring Network Chain Entity for Real-Time Multiple ID Detection at Scale in Fantasy Sports.** AIMLSystems '23, ACM, Article 26. [doi.org/10.1145/3639856.3639882](https://doi.org/10.1145/3639856.3639882)

</div>
</div>

<div class="cv-entry">
<div class="cv-entry__when">2023</div>
<div class="cv-entry__what" markdown="1">

**Maximizing Success Rate of Payment Routing using Non-stationary Bandits.** AIMLSystems '23, ACM, Article 27. [doi.org/10.1145/3639856.3639883](https://doi.org/10.1145/3639856.3639883)

</div>
</div>

<div class="cv-entry">
<div class="cv-entry__when">2023</div>
<div class="cv-entry__what" markdown="1">

**Accelerating Causal Algorithms for Industrial-scale Data: A Distributed Computing Approach with Ray.** AIMLSystems '23, ACM, Article 24. [doi.org/10.1145/3639856.3639880](https://doi.org/10.1145/3639856.3639880)

</div>
</div>

{% if site.show_talks %}
## Talks

See the [Talks]({{ '/talks/' | relative_url }}) section.
{% endif %}

{% if site.show_teaching %}
## Teaching

See the [Teaching]({{ '/teaching/' | relative_url }}) section.
{% endif %}

## Contact

- **GitHub** - [github.com/nilesh-patil](https://github.com/nilesh-patil)
- **LinkedIn** - [linkedin.com/in/ensembledme](https://www.linkedin.com/in/ensembledme)
- **Google Scholar** - [scholar.google.co.in/citations?user=IIabY1sAAAAJ](https://scholar.google.co.in/citations?user=IIabY1sAAAAJ)
- **Medium** - [nilesh-patil.medium.com](https://nilesh-patil.medium.com/)
