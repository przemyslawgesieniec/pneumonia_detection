import models.NeuralNetworkModelConfiguration;
import org.apache.log4j.BasicConfigurator;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.Random;

public class PneumoniaDetectionTests {

    private final static Logger log = LoggerFactory.getLogger(PneumoniaDetectionTests.class);
    private final static String TEST_PATH = "test";
    private final static String TRAIN_PATH = "train";

    private final static int HEIGHT = 224;
    private final static int WIDTH = 224;
    private final static int CHANNELS = 3;
    private final static int BATCH_SIZE = 10;
    private final static int EPOCHS = 10;
    private final static int SEED = 123;
    private final static int POSSIBLE_LABEL_NUMBER = 3;

    public static void main(String[] args) throws IOException, InterruptedException {

        BasicConfigurator.configure();
//        org.apache.log4j.PropertyConfigurator.configure("log4j.properties");

        final Random rng = new Random(SEED);

        log.info("----------DATA SETUP-----------");

        ClassLoader loader = PneumoniaDetectionTests.class.getClassLoader();
        final File trainData = new File(Objects.requireNonNull(loader.getResource(TEST_PATH)).getFile());
        final File testData = new File(Objects.requireNonNull(loader.getResource(TRAIN_PATH)).getFile());

        final FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, rng);
        final FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, rng);

        final ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        final ImageRecordReader imageRecordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);

        imageRecordReader.initialize(train);

        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE, 1, POSSIBLE_LABEL_NUMBER);

        log.info("----------MODEL-----------");

        NeuralNetworkModelConfiguration neuralNetConfiguration = new NeuralNetworkModelConfiguration(HEIGHT, WIDTH, CHANNELS, SEED, POSSIBLE_LABEL_NUMBER);

        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(neuralNetConfiguration.getAlexNet());
        multiLayerNetwork.init();

        multiLayerNetwork.setListeners(new ScoreIterationListener(10));

//        applyWebUI(multiLayerNetwork);

        log.info("----------TRAINING-----------");
        multiLayerNetwork.fit(dataSetIterator, EPOCHS);

        log.info("----------EVALUATE TRAINED MODEL-----------");

        Evaluation evaluation = new Evaluation(POSSIBLE_LABEL_NUMBER);

        while (dataSetIterator.hasNext()) {
            final DataSet next = dataSetIterator.next();
            INDArray output = multiLayerNetwork.output(next.getFeatures());
            evaluation.eval(next.getLabels(), output);
        }
        log.info(evaluation.stats(false, true));

//        log.info("\n" + evaluation.stats());

        evaluateModel(dataSetIterator, multiLayerNetwork);

        log.info("----------CLASSIFICATION OF TEST DATA-----------");

        imageRecordReader.reset();
        imageRecordReader.initialize(test);
        dataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE, 1, POSSIBLE_LABEL_NUMBER);

        final List<String> labels = imageRecordReader.getLabels();

        ClassificationResult classificationResult = new ClassificationResult(labels, multiLayerNetwork, dataSetIterator);

        classificationResult.printClassificationStatus();
    }

    private static void applyWebUI(final MultiLayerNetwork multiLayerNetwork) {
        UIServer uiServer = UIServer.getInstance();

        StatsStorage statsStorage = new InMemoryStatsStorage();

        uiServer.attach(statsStorage);

        multiLayerNetwork.setListeners(new StatsListener(statsStorage));
    }

    private static void evaluateModel(final DataSetIterator dataSetIterator,
                                      final MultiLayerNetwork multiLayerNetwork) {


    }

}
