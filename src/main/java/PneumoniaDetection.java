import models.NeuralNetworkModelConfiguration;
import org.apache.log4j.BasicConfigurator;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.evaluation.EvaluationTools;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.util.List;
import java.util.Map;

import static utils.DataLoader.loadDataFiles;

public class PneumoniaDetection {

    private final static Logger log = LoggerFactory.getLogger(PneumoniaDetection.class);

    private final static int HEIGHT = 224;
    private final static int WIDTH = 224;
    private final static int CHANNELS = 1;
    private final static int BATCH_SIZE = 15;
    private final static int EPOCHS = 80;
    private final static int SEED = 123;
    private final static int LABELS = 3;

    private final static String TEST = "test";
    private final static String TRAIN = "train";


    public static void main(String[] args) throws Exception {
        detect();
    }

    public static void detect() throws Exception {
        BasicConfigurator.configure();

        log.info("--------LOAD DATA--------");
        final Map<String, FileSplit> data = loadDataFiles();

        final ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        final ImageRecordReader imageRecordReader =
                new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
        imageRecordReader.initialize(data.get(TRAIN));

        final DataSetIterator trainDataSetIterator =
                new RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE, 1, LABELS);

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainDataSetIterator);
        trainDataSetIterator.setPreProcessor(scaler);

        log.info("--------BUILD MODEL--------");
        final NeuralNetworkModelConfiguration neuralNetworkModelConfiguration
                = new NeuralNetworkModelConfiguration(HEIGHT, WIDTH, CHANNELS, SEED, LABELS);

        final MultiLayerNetwork multiLayerNetwork =
                new MultiLayerNetwork(neuralNetworkModelConfiguration.getCustomNet());

        applyWebUI(multiLayerNetwork);

        log.info("--------TRAIN MODEL--------");
        for (int i = 0; i < EPOCHS; i++) {
            log.info("Epoch " + i + " started");

            multiLayerNetwork.fit(trainDataSetIterator, EPOCHS);
        }

        log.info("--------EVALUATE MODEL--------");
        imageRecordReader.reset();

        imageRecordReader.initialize(data.get(TEST));
        DataSetIterator testDataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE, 1, LABELS);

        log.info(imageRecordReader.getLabels().toString());

        final Evaluation eval = new Evaluation(LABELS);
        final ROCMultiClass roc = new ROCMultiClass();

        multiLayerNetwork.doEvaluation(testDataSetIterator, eval, roc);

        log.info(eval.stats());
        EvaluationTools.exportRocChartsToHtmlFile(roc, new File("rocPlotCustomNet80.html"));

        log.info("--------CLASSIFY--------");

        imageRecordReader.reset();
        imageRecordReader.initialize(data.get(TEST));
        testDataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE, 1, LABELS);

        final List<String> labels = imageRecordReader.getLabels();

        final ClassificationResult classificationResult = new ClassificationResult(labels, multiLayerNetwork, testDataSetIterator);

        classificationResult.printClassificationStatus();
    }

    private static void applyWebUI(final MultiLayerNetwork multiLayerNetwork) {
        final UIServer uiServer = UIServer.getInstance();
        final StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        multiLayerNetwork.setListeners(new StatsListener(statsStorage));
    }

}
