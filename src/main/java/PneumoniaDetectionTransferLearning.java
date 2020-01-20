import models.NeuralNetworkModelConfiguration;
import org.apache.log4j.BasicConfigurator;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.evaluation.EvaluationTools;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.util.List;
import java.util.Map;

import static utils.DataLoader.loadDataFiles;

public class PneumoniaDetectionTransferLearning {

    private final static Logger log = LoggerFactory.getLogger(PneumoniaDetectionTransferLearning.class);

    private final static int HEIGHT = 224;
    private final static int WIDTH = 224;
    private final static int CHANNELS = 3;
    private final static int BATCH_SIZE = 100;
    private final static int EPOCHS = 1;
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
        final ImageRecordReader imageRecordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);
        imageRecordReader.initialize(data.get(TRAIN));

        final DataSetIterator trainDataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE, 1, LABELS);
        trainDataSetIterator.setPreProcessor(new VGG16ImagePreProcessor());

        log.info("--------BUILD MODEL--------");
        final NeuralNetworkModelConfiguration neuralNetworkModelConfiguration
                = new NeuralNetworkModelConfiguration(HEIGHT, WIDTH, CHANNELS, SEED, LABELS);
        final ComputationGraph vggNetConfig = neuralNetworkModelConfiguration.getVggNetConfig();

//        TransferLearningHelper transferLearningHelper =
//                new TransferLearningHelper(vggNetConfig);

//        applyWebUI(transferLearningHelper);
//        applyWebUI(vggNetConfig);
//        log.info(transferLearningHelper.unfrozenGraph().summary());

        log.info(vggNetConfig.summary());

        log.info("--------TRAIN MODEL--------");

//        for (int i = 0; i < EPOCHS; i++) {
//            transferLearningHelper.fitFeaturized(trainDataSetIterator);
//        }
        vggNetConfig.fit(trainDataSetIterator, EPOCHS);

        log.info("--------EVALUATE MODEL--------");
        imageRecordReader.reset();
        imageRecordReader.initialize(data.get(TEST));
        DataSetIterator testDataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE, 1, LABELS);
        testDataSetIterator.setPreProcessor(new VGG16ImagePreProcessor());

        log.info(imageRecordReader.getLabels().toString());

        final Evaluation eval = new Evaluation(LABELS);
        final ROCMultiClass roc = new ROCMultiClass();

//        transferLearningHelper.unfrozenGraph().doEvaluation(testDataSetIterator, eval, roc);
        vggNetConfig.doEvaluation(testDataSetIterator, eval, roc);

        log.info(eval.stats());
        EvaluationTools.exportRocChartsToHtmlFile(roc, new File("rocPlotVGG.html"));

        log.info("--------CLASSIFY--------");
        final List<String> labels = imageRecordReader.getLabels();

//        modelPredict(vggNetConfig,testDataSetIterator);

//        final ClassificationResult classificationResult = new ClassificationResult(labels, transferLearningHelper.unfrozenGraph().pr, testDataSetIterator);
//        modelPredict(transferLearningHelper.unfrozenGraph(), testDataSetIterator);
//        classificationResult.printClassificationStatus();
    }

//    private static void applyWebUI(final TransferLearningHelper transferLearningHelper) {
//        final UIServer uiServer = UIServer.getInstance();
//        final StatsStorage statsStorage = new InMemoryStatsStorage();
//        uiServer.attach(statsStorage);
//        transferLearningHelper.unfrozenGraph().setListeners(new StatsListener(statsStorage));
//    }

    private static void applyWebUI(final ComputationGraph vggNet) {
        final UIServer uiServer = UIServer.getInstance();
        final StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        vggNet.setListeners(new StatsListener(statsStorage));
    }

//    private static void modelPredict(ComputationGraph model,
//                                     DataSetIterator iterator) {
//        int sumCount = 0;
//        int correctCount = 0;
//
//        while (iterator.hasNext()) {
//            final DataSet next = iterator.next();
//            INDArray[] output = model.output(next.getFeatures());
//            final INDArray labels = next.getLabels();
//            int dataNum = Math.min(BATCH_SIZE, output[0].rows());
//            for (int dataIndex = 0; dataIndex < dataNum; dataIndex++) {
//                StringBuilder reLabel = new StringBuilder();
//                StringBuilder peLabel = new StringBuilder();
//                INDArray preOutput;
//                INDArray realLabel;
//                for (int digit = 0; digit < 6; digit++) {
//                    preOutput = output[digit].getRow(dataIndex);
//                    peLabel.append(Nd4j.argMax(preOutput, 1).getInt(0));
//                    realLabel = labels.getRow(dataIndex);
//                    reLabel.append(Nd4j.argMax(realLabel, 1).getInt(0));
//                }
//                boolean equals = peLabel.toString().equals(reLabel.toString());
//                if (equals) {
//                    correctCount++;
//                }
//                sumCount++;
//                log.info("real image {}  prediction {} status {}", reLabel.toString(), peLabel.toString(), equals);
//            }
//        }
//        iterator.reset();
//        System.out.println("validate result : sum count =" + sumCount + " correct count=" + correctCount);
//    }
}
