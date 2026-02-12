# Requirements Elicitation Report

## Introduction

This document details the requirements elicitation process. The goal is to understand the needs and challenges of our target audience, ensuring that we build a tool that delivers real value.

For this analysis, we employed a combination of two agile techniques: **User Journey Mapping** and **Brainstorming**. The former helped us identify the problems, while the latter helped us generate solutions (features).

---

## Technique 1: User Journey Mapping

### Process Description

To guide development, we utilized the User Journey Mapping technique, which relies on creating scenarios to visualize the end-to-end user experience. This method allowed us to map the actions, emotions, and frustrations of a typical user when interacting with the problem our tool aims to solve, revealing clear opportunities for creating high-value features.

### Persona Profile

To make the journey concrete, we created a persona representing our primary target audience.

- **Name:** Ana Costa
- **Age:** 22
- **Occupation:** Undergraduate Economics Student.
- **Goal:** Collect, visualize, and compare socioeconomic data (GDP, internet usage, etc.) from different countries for her thesis. She also wants to generate a simple forecast to strengthen her arguments but lacks advanced knowledge in Machine Learning.
- **Frustrations:** Wastes significant time navigating government data portals, downloading, and cleaning spreadsheets. Finds the process of training ML models intimidating and struggles to interpret their predictions.

### Scenario (Journey Goal)

Ana needs to compare the evolution of GDP and the percentage of internet usage between Brazil and Canada for her thesis. Additionally, she wants to generate a GDP forecast for the upcoming year to include in her trend analysis.

### Evidence (Journey Map)

The map below represents Ana's journey. The identified opportunities served as the raw material for the brainstorming session.


| Journey Stages | Discovery and Access | Data Selection and Visualization | Model Training | Results Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **Story** | Ana hears about a new data analysis tool and decides to test it out. | With the tool open, Ana selects the countries and indicators she needs for her research. | Curious about the forecasting function, Ana decides to train a model to estimate next year's GDP. | The app displays the prediction and the performance metrics of the trained model. |
| **Actions** | Navigates to the application link. | 1. Selects "Brazil" and "Canada".<br>2. Chooses indicators (GDP, etc.).<br>3. Observes the generated charts. | 1. Navigates to the ML tab.<br>2. Chooses a model (e.g., Linear Regression).<br>3. Clicks the "Train Model" button. | 1. Reads the predicted value.<br>2. Checks metrics (MAE, R², etc.).<br>3. Analyzes the "Predicted vs. Actual" chart. |
| **Touchpoints** | Streamlit Home Page. | Configuration Sidebar and the "Data Exploration" tab. | Sidebar and the train button in the "Modeling" tab. | Result containers in the "Modeling" tab. |
| **Emotions** | 🤔 Curious | 😊 Satisfied | 😬 Apprehensive | 🤯 Confused / 😄 Impressed |
| **Pain Points** | "Is this reliable? Where does the data come from?" | "I wish I could compare two indicators on the same chart." | "Which model should I choose? I don't understand the difference between them." | "What does 'R² = 0.85' mean? How does each data point impact the model's prediction?" |
| **Opportunities (Backstage Actions)** | Display the data source (World Bank) and the last update date. | Create a comparative chart with multiple axes (dual-axis). | Add tooltips or help text explaining each model in simple terms. | Present metrics with explanatory text and include explainability techniques (XAI). |

---

## Technique 2: Brainstorming

### Process Description

After mapping the journey and identifying Ana's pain points, we held a brainstorming session to generate feature ideas. The session focused on the following guiding question: "How can we transform Ana's pain points (complexity, lack of trust, and difficulty in interpretation) into features that make our tool powerful, intuitive, and reliable?". The ideas generated were then grouped into themes, which will become our Epics.

### Evidence (Brainstorming Results)

The structure below represents the outcome of our brainstorming session, with ideas clustered by theme.

#### Theme 1: Data Analysis and Visualization

- **Ideas:**
  - Allow selection of multiple countries for side-by-side comparison.
  - Allow plotting of two different indicators on the same chart, using distinct Y-axes.
  - Add a button to "Export Chart as PNG".
  - Display the data source and the date of the last update.
  - Add a scale selector for the charts (Linear vs. Log).

#### Theme 2: Uncomplicated Machine Learning

- **Ideas:**
  - Add a help icon `(?)` next to each model with a simple explanation of how it works.
  - In addition to metrics, show a textual interpretation of performance (e.g., "This model fit the test data well.").
  - Display a "Feature Importance" chart to show what influenced the prediction the most.
  - Allow the user to adjust the train/test split percentage (e.g., 80/20, 70/30).

#### Theme 3: Dashboard and Usability

- **Ideas:**
  - Create a "Report" tab/section that summarizes all selections and results for easy screenshotting.
  - Add a button to "Export Table Data as CSV".
  - Implement a "Presentation Mode" that hides menus and leaves only charts and results visible.
  - Cache the user's last selection (country, model) in the browser for their next visit.

## Elicitation Conclusion

The combination of User Journey Mapping and Brainstorming techniques was effective. We managed to start from a realistic usage scenario, identify concrete frustrations, and translate them into a rich set of feature ideas. This process ensures that our backlog is not just a list of technical tasks, but an action plan oriented toward generating value for our persona, Ana. The ideas grouped by themes were fundamental for creating Epics and User Stories to develop the project.

---
