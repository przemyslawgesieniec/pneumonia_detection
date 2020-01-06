import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import java.util.HashMap;
import java.util.List;

public class ClassificationResult {


    private List<String> labels;
    private MultiLayerNetwork multiLayerNetwork;
    private DataSetIterator dataSetIterator;

    private final HashMap<String, Integer> correctlyClassifiedMap = new HashMap<>();
    private final HashMap<String, Integer> totalClassifiedMap = new HashMap<>();
    private final HashMap<String, Double> classificationResults = new HashMap<>();


    public ClassificationResult(final List<String> labels,
                                 final MultiLayerNetwork multiLayerNetwork,
                                 final DataSetIterator dataSetIterator) {
        this.labels = labels;
        this.multiLayerNetwork = multiLayerNetwork;
        this.dataSetIterator = dataSetIterator;
        prepareClassificationMaps();
        performClassification();
    }

    private void prepareClassificationMaps() {
        labels.forEach(label -> {
            correctlyClassifiedMap.put(label, 0);
            totalClassifiedMap.put(label, 0);
            classificationResults.put(label, 0D);
        });
    }

    public HashMap<String, Integer> getCorrectlyClassifiedMap() {
        return correctlyClassifiedMap;
    }

    public HashMap<String, Integer> getTotalClassifiedMapClassifiedMap() {
        return totalClassifiedMap;
    }

    public HashMap<String, Double> getClassificationResults() {
        return classificationResults;
    }

    private void performClassification() {
        while (dataSetIterator.hasNext()) {
            DataSet testDataSet = dataSetIterator.next();
            int labelIndex = testDataSet.getLabels().argMax(1).getInt(0);
            final int[] predicted = multiLayerNetwork.predict(testDataSet.getFeatures());
            String expectedResult = labels.get(labelIndex);
            String modelPrediction = labels.get(predicted[0]);

            if (expectedResult.equals(modelPrediction)) {
                correctlyClassifiedMap.merge(expectedResult, 1, Integer::sum);
            }
            totalClassifiedMap.merge(expectedResult, 1, Integer::sum);
        }
        labels.forEach(label -> {
            final Double correct = Double.valueOf(correctlyClassifiedMap.get(label));
            final Double total = Double.valueOf(totalClassifiedMap.get(label));
            classificationResults.put(label, correct / total);
        });
    }

    public void printClassificationStatus() {
        labels.forEach(label -> {
            System.out.printf("[%s] classified correctly %d times out of %d - %f \n",
                    label,
                    correctlyClassifiedMap.get(label),
                    totalClassifiedMap.get(label),
                    classificationResults.get(label));
        });
    }
}
