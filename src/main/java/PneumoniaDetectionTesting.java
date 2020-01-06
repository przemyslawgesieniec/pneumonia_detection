import models.NeuralNetworkModelConfiguration;
import org.apache.log4j.BasicConfigurator;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
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
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.Random;

public class PneumoniaDetectionTesting {

    private final static Logger log = LoggerFactory.getLogger(PneumoniaDetectionTesting.class);
    private final static String TEST_PATH = "test";
    private final static String TRAIN_PATH = "train";

    private final static int HEIGHT = 224;
    private final static int WIDTH = 224;
    private final static int CHANNELS = 1;
    private final static int BATCH_SIZE = 10;
    private final static int EPOCHS = 50; //should be data set size/batch size 50 works fine
    private final static int SEED = 123;
    private final static int LABELS = 3;


    public static void main(String[] args) throws IOException, InterruptedException {

        BasicConfigurator.configure();
        Random randNumGen = new Random(SEED);

        // Define the File Paths

        ClassLoader loader = PneumoniaDetectionTesting.class.getClassLoader();
        final File trainData = new File(Objects.requireNonNull(loader.getResource(TEST_PATH)).getFile());
        final File testData = new File(Objects.requireNonNull(loader.getResource(TRAIN_PATH)).getFile());

        final FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        final FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        // Extract the parent path as the image label
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        ImageRecordReader imageRecordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);

        // Initialize the record reader
        // add a listener, to extract the name
        imageRecordReader.initialize(train);
        //imageRecordReader.setListeners(new LogRecordListener());

        // DataSet Iterator
        DataSetIterator dataIter = new RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE, 1, LABELS);

        // Scale pixel values to 0-1
//        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
//        scaler.fit(dataIter);
//        dataIter.setPreProcessor(scaler);


        NeuralNetworkModelConfiguration neuralNetworkModelConfiguration
                = new NeuralNetworkModelConfiguration(HEIGHT, WIDTH, CHANNELS, SEED, LABELS);

        // Build Our Neural Network
        log.info("-------------BUILD MODEL-------------");
        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(neuralNetworkModelConfiguration.getLeNet());


        applyWebUI(multiLayerNetwork);


        log.info("TRAIN MODEL");
        for (int i = 0; i < EPOCHS; i++) {
            multiLayerNetwork.fit(dataIter);
        }

        log.info("EVALUATE MODEL");
        imageRecordReader.reset();
        imageRecordReader.initialize(test);
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE, 1, LABELS);
//        scaler.fit(dataSetIterator);
//        dataSetIterator.setPreProcessor(scaler);

        log.info(imageRecordReader.getLabels().toString());

        // Create Eval object with 10 possible classes
        Evaluation eval = new Evaluation(LABELS);

        // Evaluate the network
        while (dataSetIterator.hasNext()) {
            DataSet next = dataSetIterator.next();
            INDArray output = multiLayerNetwork.output(next.getFeatures());
            // Compare the Feature Matrix from the multiLayerNetwork
            // with the labels from the RecordReader
            eval.eval(next.getLabels(), output);
        }

        log.info(eval.stats());

        log.info("----------CLASSIFICATION OF TEST DATA-----------");

        imageRecordReader.reset();
        imageRecordReader.initialize(test);
        dataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, PneumoniaDetectionTesting.BATCH_SIZE, 1, PneumoniaDetectionTesting.LABELS);

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

}
