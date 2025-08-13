# AcademicJournalClassifier

A project for classifying academic journal articles using data mining and warehousing techniques.

# 1. Business Understanding

## Problem Statement

The University of Zambia (UNZA) hosts a growing repository of academic journal articles across multiple disciplines. However, these articles are not systematically categorized according to Zambia’s Vision 2030 development sectors. This lack of alignment presents a missed opportunity to leverage UNZA’s intellectual output for national strategic planning, policy formulation, and sectoral development monitoring.

This project aims to develop a data-driven classification system that maps UNZA journal articles to the appropriate Vision 2030 sectors using machine learning techniques. By automating this classification, we intend to bridge the gap between academic research and national development priorities, enabling policymakers, researchers, and institutions to better identify and track sectoral contributions and trends.

## Objectives

1. **Align UNZA’s research with national priorities:**  
    Systematically map academic journal articles to Zambia’s Vision 2030 development sectors to highlight how UNZA’s intellectual output contributes to national development goals.

2. **Enable evidence-based decision-making:**  
    Provide policymakers, researchers, and development stakeholders with an accessible, data-driven tool for identifying sectoral trends and gaps in research, supporting targeted policy formulation and strategic resource allocation.

3. **Automate and scale research classification:**  
    Develop a machine learning–powered system to efficiently classify and update research article categorization, ensuring scalability as UNZA’s repository grows and enabling continuous monitoring of sectoral contributions.

## Data Mining Goals

1. **Design a supervised multi-class classification model**  
    Assign each UNZA journal article to one of Zambia’s Vision 2030 sectors based on metadata (title, abstract, keywords).

    - *Purpose*: Reveal alignment between academic output and national development areas.
    - *Method*: Use labeled training data mapped to Vision 2030 sectors, extracted from a subset of articles.
    - *Expected Output*: Accurate labels such as “Education,” “Agriculture,” “Health,” “Infrastructure,” etc.

2. **Identify latent research clusters and anomalies**  
    Use unsupervised learning (e.g., clustering or topic modeling) to uncover emerging themes or neglected areas.

    - *Purpose*: Help decision-makers identify new or missing areas of national interest not currently emphasized in the Vision 2030 framework.
    - *Method*: Apply techniques like K-Means, DBSCAN, or LDA topic modeling on text embeddings.
    - *Expected Output*: Visual or descriptive reports of discovered themes or outliers.

3. **Deploy a scalable, retrainable classification pipeline**  
    Use modern ML techniques and modular design.

    - *Purpose*: Automate the tagging process for future UNZA research uploads.
    - *Method*: Build a modular pipeline for preprocessing, vectorization (e.g., TF-IDF or BERT), training, evaluation, and inference.
    - *Expected Output*: A script or web app that classifies new articles on upload.

4. **Continuously evaluate model performance**  
    Use metrics such as F1-score, accuracy, and confusion matrices.

    - *Purpose*: Ensure system reliability and adaptiveness as language and research topics evolve.
    - *Method*: Establish a validation framework and regularly benchmark models.
    - *Expected Output*: Monitoring logs or retraining criteria to prevent model drift.

## Initial Project Success Criteria

The project will be considered initially successful if the supervised classification model achieves at least **60% accuracy** in assigning UNZA journal articles to the correct Zambia Vision 2030 development sectors.

This baseline is realistic for a first iteration, considering:

- Data quality issues (e.g., incomplete or inconsistent titles, abstracts, or keywords)
- Sector overlap, where some research spans multiple development areas
- Model maturity, as this is the initial deployment and will improve with further training and tuning

Achieving this baseline will:

- Demonstrate that the model performs significantly above random guessing
- Provide policymakers and researchers with a usable starting point for tracking sectoral research contributions
- Establish a functional foundation for refining the system toward higher accuracy and more adoption
