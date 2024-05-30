
# Documentation - Proof of Concept Dataset 
Dasboard consortium meeting 17-07-2024

## Abstract 
### Background
This section will contain the abstract written by Bruno, highlighting the goal of this dashboard including some research questions to make the mini project objective. 

### Research questions 
1. What are average patient age/operation duration/number of open sternums for different surgeries. (perhaps not clinically relevant, but showing   what is possible and it does allow for comprison between real and synthetic data)  
2. How do patients compare to the average patient for different procedures? 
In other words: how does 1 singular (e.g. currently admitted patient) compare to the entire cohort? 
    - What is the duration of the intervention compared to average intervention duration?
    - What is the average ECMO duration (heart-lung machine usage), compared to average cohort?
    - What is the bypass time compared to the cohort?
    - ~~What is the age/weight/length compared to average of the cohort?~~
    - Possible additional: What is the average intubation time after intervention A/B?


## Data 

Example: 
||identifier_value| subject_patient_value   |      time_OR      |  postop_diagnosis_code |  postop_diagnosis_text   |         procedure_code      |  procedure_text |   procedure_duration    | procedure_age | postop_status_sternum  | ECC_duration| AoX_duration | DHCA_duration   | ACP_duration
|---|---|----------|-------------|------|----------|-------------|------|------|-----|---|--|--|--|---|
|Example|| 23ea55c2b41d40c3b222d638fa287a4x | 840       |  10572  | atriumseptumdefect in fossa ovalis |333040|Hart - atriumseptumdefect sluiten -asd ii-|228|5859|1|97|45|0|NULL|
|Units||n.a.|minutes|n.a.|n.a.|n.a.|n.a.|minutes|days|open/closed(1/0)|minutes|minutes|minutes|minutes|
|Dtype||string|integer|integer|string|string|string|integer|integer|integer|integer|integer|integer|integer|
|NaN characteristics||n.a.|n.a.|not filled in = missing|not filled in = missing|n.a.|n.a.|negative = registration error| n.a. |n.a.|not filled in = not used for that procedure|not filled in = not used for that procedure|not filled in = not used for that procedure|not filled in = not used for that procedure or not filled in correctly? 

|Word/abbreviation| Meaning|
|---|------|
|EHR|Electroninc Health Record|
|Intervention   | surgery/operation, = admission to an operation room. |
|Procedure      | One intervention can consist of multiple procedure. <br> Within our EHR the surgeon has to select 1 as the main procedure.|
|SubjectID| Anonymized patient-ID |
|operationID| Anonymized operation-ID |
|ECC_duration| Cumulative time that patient had a heart-lung machine during the intervention|
|AoX_duration| Cumulative Aortic clambing time during the intervention|
|DHCA-duration| Deep Hypothermic Cardiac Arrest = duration of cardiac arrest while to lowering body temperature to reduce tissue damage|
|ACP_duration| Antegrade Cerebral perfusion duration = blood delivery to the brain using specific insertions from the heart-lung machine|
<!-- |PatientID| Un-anonymized patient-ID| -->


### Remarks:
Originally, the idea was to only display the 'main procedure' for each intervention recorded in the sytem. However, in at least 60 interventions, the insertion of the heart-lung machine was selected as the main procedure, which is clinically irrelevant. 
Therefore, all 'surgical procedures' are listed for each intervention and an identifier is added. 
For the proof of concept dashboard, we could choose to  use the 'simple version' of the dataset. 

#### Advantages: 
- More specific procedures can be compared = clincally relevant.

#### Disadvantages
- Anoter identifier was added. 
- When comparing different procedures, one patient can end up wihtin both groups if data analysis does not take subject ID into account.   
- Creating a (more) complex dataset where one line does not equal one intervention.  

#### Other remarks :
Furthermore, all surgeries containing a procedure that occurred less than 5 times were removed, similarly the surgeries with post-operative diagnoses that occurred less than 5 times were removed . 
- We should determine the preferred data-types. 
- Patient ID is anonymized using the following python package: [uuid4](https://docs.python.org/3/library/uuid.html#uuid.UUID.hex). It uses the function: `uuid4().hex`
- Operationnumber (/ID) is anonymised using the same python package.
- In some cases, the `procedure_duration` is missing a start- or end-time, this sometimes results in a negative value. The best alternative is looked for. 
    - Perhaps this is not a problem that should be fixed for what we want to showcase with this POC. 
- `ACP_duration` is usually NaN, which might change in the future. Currently, I am not sure wheter it is not filled in or is not not used during the operation. The 'raw data' for this parameter is available, however it is not (yet) implemented in a way I can link it to an intervention. (I suggest leaving the parameter out for now!)
- A similar problem (linking data to the interventions) exists for the `height` and `weight`. A 'workaround' for this is easier and currently worked on.   
- During our last meeting, we discussed whether 'patterns' exists within the data (like surgery B always follows surgery A). My answer was no, however after further inspecition, there are some surgeries that do occur in a somwhat fixed structure (might have extra interventions in between). An example is procedure codes: `333021A` & `333024B`.
 


I will keep the keyfiles seperate and savely stored. If there is any need to check a few values, I can still look in the database for additional information. 