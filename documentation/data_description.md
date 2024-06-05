
# Documentation - Proof-of-Concept Dataset

Dashboard consortium meeting 17-07-2024.

## Abstracts

### Background

This section will contain the abstract written by Bruno, highlighting the goal of this dashboard including some research questions to make the mini project objective.

### Research questions

1. What are average patient age/operation duration/number of open sternums for different surgeries. (perhaps not clinically relevant, but it is showing the possibilities and it does allow for comparison between real and synthetic data)
2. How do patients compare to the average patient for different procedures?
In other words: how does 1 singular (e.g. currently admitted patient) compare to the entire cohort?
    - What is the duration of the intervention compared to cohort's average procedure duration?
    - What is the average ECMO duration (heart-lung machine usage), compared to average cohort?
    - What is the bypass time compared to the cohort?
    - What is the age ~~/weight/length~~ compared to average of the cohort?
    - ~~Possible additional: What is the average intubation time after intervention A/B?~~

## Data

Example:
||surgery_id| subject_id   |  postop_diagnosis_code |  postop_diagnosis_text   |         procedure_code      |  procedure_text | main-procedure|  procedure_duration    | procedure_age | postop_status_sternum  | ECC_duration| AoX_duration | DHCA_duration   | ACP_duration
|---|--------|-------------|------|----------|-----|--------|------|------|-----|---|--|--|--|---|
|*Example*|23ea55c2b41d40c3b222d638fa287a4x| 23ea55c2b41d40c3b222d638fa287a4x |  10572  | atriumseptumdefect in fossa ovalis |333040|Hart - atriumseptumdefect sluiten -asd ii-|1|228|5859|1|97|45|0|0|
|*Units*|n.a.|n.a.|n.a.|n.a.|n.a.|n.a.|yes/no (1/0)|minutes|days|open/closed (1/0)|minutes|minutes|minutes|minutes|
|*Dtype*|object|object|float64|object|object|object|integer|integer|integer|integer|integer|integer|integer|integer|

|Word/abbreviation| Meaning|
|---|------|
|EHR|Electroninc Health Record|
|Surgery   | intervention/operation, = admission to an operation room. |
|Procedure      | One surgery can consist of multiple procedure. Within our EHR the surgeon has to select 1 as the main procedure.|
|subject_id| Pseudonymized patient-ID |
|surgery_id| Pseudonymized surgery-ID |
|ECC_duration| Cumulative time that patient had a heart(-lung) machine during the intervention|
|AoX_duration| Cumulative Aortic clamping time during the intervention|
|DHCA-duration| Deep Hypothermic Cardiac Arrest = duration of cardiac arrest while lowering body temperature to reduce tissue damage|
|ACP_duration| Antegrade Cerebral Perfusion duration = blood delivery to the brain using specific insertions from the heart-lung machine|

### Remarks

Originally, the idea was to only display the 'main procedure' for each surgery recorded in the system. However, in at least 60 interventions, the insertion of the heart-lung machine was selected as the main procedure, which most often is clinically irrelevant.
Therefore, all 'surgical procedures' are listed for each surgery. An identifier is added (`surgery_id`) and the column `main_procedure` was introduced.
For the proof-of-concept dashboard, we could choose to use the 'simple version' of the dataset.

#### Advantages

- More specific procedures can be compared = clinically relevant.

#### Disadvantages

- Another identifier was added.
- When comparing different procedures, one patient can end up within both groups if data analysis does not take subject ID into account.
- Creating a (more) complex dataset where one line does not equal one intervention.

#### Other remarks

Furthermore, all surgeries containing a procedure that occurred less than 5 times were removed, similarly the surgeries with post-operative diagnoses that occurred less than 5 times were removed.

- We should determine the preferred datatypes. (I've created a suggestion in the data-overview)
- Patient ID is anonymized using the following python package: [uuid4](https://docs.python.org/3/library/uuid.html#uuid.UUID.hex). It uses the function: `uuid4().hex`.
- Surgery_id is anonymised using the same python package.
- In some cases, the `procedure_duration` is missing a start- or end-time, this sometimes results in a negative value. The best alternative is looked for
    - I've created the first temporary solution, which is not perfect.
    - Perhaps this is not a problem that should be fixed for what we want to showcase with this Proof-of-Concept.
- `ACP_duration` is usually quite often not available. For creating synthetic data, this might not be the best parameter to include. We could drop the parameter if needed.
- Linking data to the interventions without adding new identifiers is difficult for the `height` and `weight` of the patient. I'm looking for a possibile work-around, but can't promise anything before the consortium meeting.
- During one of our meetings, we discussed whether 'patterns' exists within the data (like surgery B always follows surgery A). My answer was no, however after further inspection, there are some surgeries that do occur in a somewhat fixed structure (might have extra interventions in between). An example is procedure codes: `333021A` & `333024B`.

I will keep the 'keyfiles' separate and safely stored. If there is any need to check a few values, I can still look in the database for additional information.
