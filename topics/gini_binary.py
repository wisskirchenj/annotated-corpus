import numpy as np

stumps = np.array([[[3,1], [3,0]], [[1,3], [2,1]]])

for clas in stumps:
    num_classes = clas.shape[0]
    print(num_classes, "classes")
    gini = 1 - np.sum((clas/clas.sum(axis=1, keepdims=True))**2, axis=1)
    print("Gini:", gini)
    gini_impurity = np.sum(gini * clas.sum(axis=1) / clas.sum())
    print(f"Gini_impurity: {gini_impurity:.2f}")
