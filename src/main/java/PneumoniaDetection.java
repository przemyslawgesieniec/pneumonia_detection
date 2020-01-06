import models.NeuralNetworkModelConfiguration;
import org.apache.log4j.BasicConfigurator;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import static utils.DataLoader.loadDataFiles;

public class PneumoniaDetection {

    private final static Logger log = LoggerFactory.getLogger(PneumoniaDetection.class);

    private final static int HEIGHT = 224;
    private final static int WIDTH = 224;
    private final static int CHANNELS = 1;
    private final static int BATCH_SIZE = 10;
    private final static int EPOCHS = 1; //should be data set size/batch size 50 works fine
    private final static int SEED = 123;
    private final static int LABELS = 3;

    private final static String TEST = "test";
    private final static String TRAIN = "train";


    public static void main(String[] args) throws IOException, InterruptedException {

        BasicConfigurator.configure();

        log.info("--------LOAD DATA--------");
        final Map<String, FileSplit> data = loadDataFiles();

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader imageRecordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
        imageRecordReader.initialize(data.get(TRAIN));

        DataSetIterator trainDataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE, 1, LABELS);

        log.info("--------BUILD MODEL--------");
        NeuralNetworkModelConfiguration neuralNetworkModelConfiguration
                = new NeuralNetworkModelConfiguration(HEIGHT, WIDTH, CHANNELS, SEED, LABELS);
        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(neuralNetworkModelConfiguration.getAlexNet());

        applyWebUI(multiLayerNetwork);

        log.info("--------TRAIN MODEL--------");
        multiLayerNetwork.fit(trainDataSetIterator, EPOCHS);


        log.info("--------EVALUATE MODEL--------");
        imageRecordReader.reset();
        imageRecordReader.initialize(data.get(TEST));
        DataSetIterator testDataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE, 1, LABELS);

        log.info(imageRecordReader.getLabels().toString());

        Evaluation eval = new Evaluation(LABELS);

        while (testDataSetIterator.hasNext()) {
            DataSet next = testDataSetIterator.next();
            INDArray output = multiLayerNetwork.output(next.getFeatures());
            eval.eval(next.getLabels(), output);
        }

        log.info(eval.stats());

        log.info("--------CLASSIFY--------");

        imageRecordReader.reset();
        imageRecordReader.initialize(data.get(TEST));
        testDataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, PneumoniaDetection.BATCH_SIZE, 1, PneumoniaDetection.LABELS);

        final List<String> labels = imageRecordReader.getLabels();

        ClassificationResult classificationResult = new ClassificationResult(labels, multiLayerNetwork, testDataSetIterator);

        classificationResult.printClassificationStatus();
    }

    private static void applyWebUI(final MultiLayerNetwork multiLayerNetwork) {
        UIServer uiServer = UIServer.getInstance();

        StatsStorage statsStorage = new InMemoryStatsStorage();

        uiServer.attach(statsStorage);

        multiLayerNetwork.setListeners(new StatsListener(statsStorage));
    }

    private static void performEvaluation() {

    }

}
