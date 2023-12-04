# Alzheimer disease classifier based on handwriting tasks
Work done on top of DARWIN dataset to predict Alzheimer Disease based on writing tasks. The target in this project is to combine different
approaches in machine learning and deep learning methodologies to analyze data from Alzheimer’s disease patients. The study focuses on two key aspects: the
analysis of pen gesture dynamics during various tasks and the examination of the same task-related images using deep learning architectures. By integrating
these approaches, a comprehensive understanding of the cognitive impairments associated with Alzheimer’s disease can be achieved.

In addition to the deep learning analysis, the project will investigate the extraction of deep features from images. These deep features will then be utilized
as inputs to machine learning classifiers, enabling a fusion of the information derived from both approaches.
The process is described as follows:
* Data Description and Acquisition}
* Feature Selection and Preprocessing
* Classifier Selection and Hyperparameter tunning
* Deep Learning Approach
* Deep Features Extraction

---
Data is obtained by asking each participant to complete a series of tasks, divided in different categories and levels of difficulty, being the next task more demanding
than the previous one in terms of the cognitive functions required to fulfill the task. Overall, there are 25 tasks divided in three groups: Memory tasks (MT), copy
and reverse copy tasks (CT) and graphic tasks (GT). All of them need to be written on A4 white sheets, which are stapled and placed on a graphic tablet which records
the movements of the pen used by the examined subject.

---
## Results for task 1
![image](https://github.com/dagazrev/alzheimer_classifier/assets/33989743/80a87ea8-ab8d-4e97-9947-3554d7592576)

There's an overall best performance using Gradient Boosting and Random forest classifiers in random forest. The performance depends on the task and the feature selection used (Random Forest with 10 or 400 estimators). For the overall performance per task, refer to the report pdf file.

Overall, this are the best results in Machine learning:
![image](https://github.com/dagazrev/alzheimer_classifier/assets/33989743/72a80966-3285-463a-93c0-edf83321440b)

### Deep learning approach:
Deep learning approach uses the image of the writing tasks. This complement the dynamics of the pen gestures in the tabular data and shows how hard for some impaired patients is to perform the simple written activities. Some tasks look like:

![image](https://github.com/dagazrev/alzheimer_classifier/assets/33989743/db59a5ea-a73d-40dd-9c10-b76da09c6010)

Transfer learning will be used. The overall pipeline looks like this:

![image](https://github.com/dagazrev/alzheimer_classifier/assets/33989743/56beb9bd-ab12-4e34-94d3-bd5e0350c760)

Here, the AUC is used as the evaluation metric and the results, after evaluating individual models and doing ensemble learning:
* Task 1: Phase 3 Ensemble of VGG19 + inceptionv3 for an AUC of 0.73.
* Task 2: Phase 3 Ensemble of inceptionV3 + inceptionResnetV2 with an AUC of 0.85.
* Task 3: Phase 2 VGG19 with an AUC of 0.90.
* Task 4: Phase 1 resnet50 with an AUC of 0.90.
* Task 9: Phase 1.5 inceptionResnetV2 with an AUC of 0.89.
* Task 10: Phase 1 inceptionResnetV2 with an AUC of 0.89.
---
DARWIN dataset: N. D. Cilia, G. De Gregorio, C. De Stefano, F. Fontanella, A. Marcelli, A. Parziale, "Diagnosing Alzheimer’s disease from on-line handwriting: A novel dataset and performance benchmarking", Engineering Applications of Artificial Intelligence, Volume 111, 2022, 104822, ISSN 0952-1976, https://doi.org/10.1016/j.engappai.2022.104822.https://www.sciencedirect. com/science/article/pii/S0952197622000902
