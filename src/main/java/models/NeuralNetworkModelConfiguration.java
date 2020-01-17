package models;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.AlexNet;
import org.deeplearning4j.zoo.model.LeNet;
import org.deeplearning4j.zoo.model.ResNet50;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.IOException;

public class NeuralNetworkModelConfiguration {


    private IUpdater updater = new AdaDelta();
    private CacheMode cacheMode = CacheMode.NONE;
    private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    private int[] inputShape;
    private int height;
    private int width;
    private int channels;
    private long seed;
    private int numClasses;

    public NeuralNetworkModelConfiguration(final int height,
                                           final int width,
                                           final int channels,
                                           final long seed,
                                           final int numLabels) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.seed = seed;
        this.numClasses = numLabels;
        this.inputShape = new int[]{channels, width, height};
    }

    public MultiLayerConfiguration getAlexNet() {
        return AlexNet.builder()
                      .seed(seed)
                      .inputShape(new int[]{channels, width, height})
                      .numClasses(numClasses)
                      .build()
                      .conf();
    }

    private ZooModel getResNet50() {
        return ResNet50.builder()
                       .inputShape(new int[]{channels, width, height})
                       .seed(seed)
                       .numClasses(numClasses)
                       .build();
    }

    private ZooModel getVGG16() {
        return VGG16.builder()
//                    .inputShape(new int[]{channels, width, height})
//                    .seed(seed)
//                    .numClasses(numClasses)
                    .build();
    }

    private FineTuneConfiguration getFineTuneConfiguration() {
        return new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(5e-5))
                .seed(seed)
                .build();
    }

    public ComputationGraph getVggNetConfig() throws IOException {

        ZooModel zooModel = getVGG16();
        ComputationGraph vgg = (ComputationGraph) zooModel.initPretrained();

        return new TransferLearning.GraphBuilder(vgg)
                .fineTuneConfiguration(getFineTuneConfiguration())
                .setFeatureExtractor("fc2")
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096).nOut(numClasses)
                                .weightInit(new NormalDistribution(0,0.2*(2.0/(4096+numClasses)))) //This weight init dist gave better results than Xavier
                                .activation(Activation.SOFTMAX).build(),
                        "fc2")
                .build();
    }

    public MultiLayerConfiguration getLeNet() {
        return LeNet.builder()
                    .seed(seed)
                    .inputShape(new int[]{channels, width, height})
                    .numClasses(numClasses)
                    .build()
                    .conf();
    }

    public MultiLayerConfiguration getSimpleCNN() {
        return new NeuralNetConfiguration
                .Builder()
                .seed(seed)
                .activation(Activation.IDENTITY)
                .weightInit(WeightInit.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(updater)
                .cacheMode(cacheMode)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .convolutionMode(ConvolutionMode.Same)
                .list()
                // block 1
                .layer(0, new ConvolutionLayer.Builder(new int[]{7, 7}).name("image_array").nIn(inputShape[0]).nOut(16).build())
                .layer(1, new BatchNormalization.Builder().build())
                .layer(2, new ConvolutionLayer.Builder(new int[]{7, 7}).nIn(16).nOut(16).build())
                .layer(3, new BatchNormalization.Builder().build())
                .layer(4, new ActivationLayer.Builder().activation(Activation.RELU).build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{2, 2}).build())
                .layer(6, new DropoutLayer.Builder(0.5).build())

                // block 2
                .layer(7, new ConvolutionLayer.Builder(new int[]{5, 5}).nOut(32).build())
                .layer(8, new BatchNormalization.Builder().build())
                .layer(9, new ConvolutionLayer.Builder(new int[]{5, 5}).nOut(32).build())
                .layer(10, new BatchNormalization.Builder().build())
                .layer(11, new ActivationLayer.Builder().activation(Activation.RELU).build())
                .layer(12, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{2, 2}).build())
                .layer(13, new DropoutLayer.Builder(0.5).build())

                // block 3
                .layer(14, new ConvolutionLayer.Builder(new int[]{3, 3}).nOut(64).build())
                .layer(15, new BatchNormalization.Builder().build())
                .layer(16, new ConvolutionLayer.Builder(new int[]{3, 3}).nOut(64).build())
                .layer(17, new BatchNormalization.Builder().build())
                .layer(18, new ActivationLayer.Builder().activation(Activation.RELU).build())
                .layer(19, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{2, 2}).build())
                .layer(20, new DropoutLayer.Builder(0.5).build())

                // block 4
                .layer(21, new ConvolutionLayer.Builder(new int[]{3, 3}).nOut(128).build())
                .layer(22, new BatchNormalization.Builder().build())
                .layer(23, new ConvolutionLayer.Builder(new int[]{3, 3}).nOut(128).build())
                .layer(24, new BatchNormalization.Builder().build())
                .layer(25, new ActivationLayer.Builder().activation(Activation.RELU).build())
                .layer(26, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{2, 2}).build())
                .layer(27, new DropoutLayer.Builder(0.5).build())


                // block 5
                .layer(28, new ConvolutionLayer.Builder(new int[]{3, 3}).nOut(256).build())
                .layer(29, new BatchNormalization.Builder().build())
                .layer(30, new ConvolutionLayer.Builder(new int[]{3, 3}).nOut(numClasses).build())
                .layer(31, new GlobalPoolingLayer.Builder(PoolingType.AVG).build())
                .layer(32, new ActivationLayer.Builder().activation(Activation.SOFTMAX).build())
                .layer(33, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .weightInit(new NormalDistribution(0, 0.005))
                        .biasInit(0.1)
                        .build())

                .setInputType(InputType.convolutional(inputShape[2], inputShape[1],
                        inputShape[0]))
                .build();
    }
}
