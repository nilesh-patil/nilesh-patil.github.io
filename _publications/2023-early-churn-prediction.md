---
title: "Early Churn Prediction from Large Scale User-Product Interaction Time Series"
collection: publications
permalink: /publications/2023-early-churn-prediction/
date: 2023-12-15
venue: "International Conference on Machine Learning and Applications (ICMLA), IEEE"
paper_url: "https://doi.org/10.1109/ICMLA58977.2023.00314"
paperurl: "https://doi.org/10.1109/ICMLA58977.2023.00314"
authors: "<strong>Nilesh Patil</strong>, et al."
citation: "<strong>Patil, N.</strong> et al. (2023). Early Churn Prediction from Large Scale User-Product Interaction Time Series. 2023 International Conference on Machine Learning and Applications (ICMLA). IEEE."
excerpt: "Deep-learning approach to predicting user churn from interaction time series at 250M+ user scale, identifying churn risk far earlier than session-aggregate baselines."
tags: [deep-learning, churn-prediction, time-series, recommender-systems]
---

A deep-learning model that ingests raw user-product interaction time series (rather than aggregated session features) to surface churn signal weeks before conventional baselines. Demonstrated on a 250M+ user feed in a production fantasy-sports setting. The architecture captures temporal cadence and ordering of events that flattened features discard, and trades off lead time vs. precision via a tunable horizon head.
