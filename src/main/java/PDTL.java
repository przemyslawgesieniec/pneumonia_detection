/* *****************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.evaluation.EvaluationTools;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import java.io.File;
import java.util.Map;

import static utils.DataLoader.loadDataFiles;

/**
 * @author susaneraly on 3/10/17.
 * <p>
 * Important:
 * Run the class "FeaturizePreSave" before attempting to run this. The outputs at the boundary of the frozen and unfrozen
 * vertices of a model are saved. These are referred to as "featurized" datasets in this description.
 * On a dataset of about 3000 images which is what is downloaded this can take "a while"
 * <p>
 * Here we see how the transfer learning helper can be used to fit from a featurized datasets.
 * We attempt to train the same model architecture as the one in "EditLastLayerOthersFrozen".
 * Since the helper avoids the forward pass through the frozen layers we save on computation time when running multiple epochs.
 * In this manner, users can iterate quickly tweaking learning rates, weight initialization etc` to settle on a model that gives good results.
 */
@SuppressWarnings("DuplicatedCode")
public class PDTL {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(PDTL.class);

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

        //Import vgg
        //Note that the model imported does not have an output layer (check printed summary)
        //  nor any training related configs (model from keras was imported with only weights and json)
        log.info("\n\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");
        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained();
        log.info(vgg16.summary());

        //Decide on a fine tune configuration to use.
        //In cases where there already exists a setting the fine tune setting will
        //  override the setting for all layers that are not "frozen".
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .updater(new Nesterovs(3e-5, 0.9))
                .seed(SEED)
                .build();

        //Construct a new model with the intended architecture and print summary
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("fc2") //the specified layer and below are "frozen"
                .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096).nOut(LABELS)
                                .weightInit(new NormalDistribution(0, 0.2 * (2.0 / (4096 + LABELS)))) //This weight init dist gave better results than Xavier
                                .activation(Activation.SOFTMAX).build(),
                        "fc2")
                .build();
        log.info(vgg16Transfer.summary());

        final Map<String, FileSplit> data = loadDataFiles();

        final ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        final ImageRecordReader imageRecordReader = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, labelMaker);

        imageRecordReader.initialize(data.get(TRAIN));
        final DataSetIterator trainDataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE, 1, LABELS);

        TransferLearningHelper transferLearningHelper = new TransferLearningHelper(vgg16Transfer);

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            while (trainDataSetIterator.hasNext()) {
                transferLearningHelper.fitFeaturized(trainDataSetIterator.next());
            }
            log.info("Epoch #" + epoch + " complete");
        }
        log.info("Model build complete");
        log.info("--------EVALUATE MODEL--------");

        imageRecordReader.reset();
        imageRecordReader.initialize(data.get(TEST));
        DataSetIterator testDataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE, 1, LABELS);
        testDataSetIterator.setPreProcessor(new VGG16ImagePreProcessor());

        log.info(imageRecordReader.getLabels().toString());

        final Evaluation eval = new Evaluation(LABELS);
        final ROCMultiClass roc = new ROCMultiClass();

        transferLearningHelper.unfrozenGraph().doEvaluation(testDataSetIterator, eval, roc);

        log.info(eval.stats());
        EvaluationTools.exportRocChartsToHtmlFile(roc, new File("rocPlotVGG.html"));
    }
}
