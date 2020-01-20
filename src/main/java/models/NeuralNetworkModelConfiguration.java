package models;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
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
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
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

//    public MultiLayerConfiguration getCustomNet() {
//        final int bias = 1;
//
//        final NeuralNetConfiguration.ListBuilder multiLayerConfiguratioListBuilder = new NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .weightInit(new NormalDistribution(0.0, 0.01))
//                .activation(Activation.RELU)
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .updater(updater)
//                .biasUpdater(new Nesterovs(2e-3, 0.9))
//                .convolutionMode(ConvolutionMode.Same)
//                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
//                .trainingWorkspaceMode(workspaceMode)
//                .inferenceWorkspaceMode(workspaceMode)
//                .cacheMode(cacheMode)
//                .l2(5 * 1e-4)
//                .miniBatch(false)
//                .list();
//
//        NeuralNetConfiguration.ListBuilder listBuilder = convBlock(multiLayerConfiguratioListBuilder, 0, 32);
//        listBuilder = convBlock(listBuilder, 1, 64);
//        listBuilder = convBlock(listBuilder, 2, 128);
//        listBuilder = convBlock(listBuilder, 3, 256);
//        listBuilder = convBlock(listBuilder, 4, 512);
//        listBuilder
//                .layer(35, new DenseLayer.Builder()
//                        .name("ffn1")
////                           .nIn(256 * 6 * 6)
//                        .nOut(4096)
//                        .weightInit(new NormalDistribution(0, 0.005))
//                        .biasInit(bias)
//                        .dropOut(0.5)
//                        .build())
//                .layer(36, new DenseLayer.Builder()
//                        .name("ffn2")
//                        .nOut(2048)
//                        .weightInit(new NormalDistribution(0, 0.005))
//                        .biasInit(bias)
//                        .dropOut(0.5)
//                        .build())
//                .layer(37, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .name("output")
//                        .nOut(numClasses)
//                        .activation(Activation.SOFTMAX)
//                        .weightInit(new NormalDistribution(0, 0.005))
//                        .biasInit(0.1)
//                        .build())
//                .setInputType(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]))
//                .build();
//
//        return listBuilder.build();
//    }

    public MultiLayerConfiguration getCustomNet() {
        final int bias = 1;

        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(updater)
                .biasUpdater(new Nesterovs(2e-2, 0.9))
                .convolutionMode(ConvolutionMode.Same)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .cacheMode(cacheMode)
                .l2(5 * 1e-4)
                .miniBatch(false)
                .list()
                //BLOCK 1
                .layer(0, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1})
                        .name("conv1b1")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Same)
                        .nIn(inputShape[0])
                        .nOut(32)
                        .build())
                .layer(1, new LocalResponseNormalization.Builder().build())
                .layer(2, new ActivationLayer.Builder().activation(Activation.RELU).build())
                .layer(3, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1})
                        .name("conv2b1")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(32)
                        .build())
                .layer(4, new BatchNormalization())
                .layer(5, new ActivationLayer.Builder().activation(Activation.RELU).build())
                .layer(6, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .padding(1, 1)
                        .name("maxpool1b1")
                        .build())
                //BLOCK 2
                .layer(7, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1})
                        .name("conv1b2")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(64)
                        .build())
                .layer(8, new BatchNormalization())
                .layer(9, new ActivationLayer.Builder().activation(Activation.RELU).build())
                .layer(10, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1})
                        .name("conv2b2")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(64)
                        .build())
                .layer(11, new BatchNormalization())
                .layer(12, new ActivationLayer.Builder().activation(Activation.RELU).build())
                .layer(13, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .padding(1, 1)
                        .name("maxpool1b2")
                        .build())

                //BLOCK 3
                .layer(14, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1})
                        .name("conv1b3")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(128)
                        .build())
                .layer(15, new BatchNormalization())
                .layer(16, new ActivationLayer.Builder().activation(Activation.RELU).build())
                .layer(17, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1})
                        .name("conv2b3")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(128)
                        .build())
                .layer(18, new BatchNormalization())
                .layer(19, new ActivationLayer.Builder().activation(Activation.RELU).build())
                .layer(20, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .padding(1, 1)
                        .name("maxpool1b3")
                        .build())

                //BLOCK 4
                .layer(21, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1})
                        .name("conv1b4")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(256)
                        .build())
                .layer(22, new BatchNormalization())
                .layer(23, new ActivationLayer.Builder().activation(Activation.RELU).build())
                .layer(24, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1})
                        .name("conv2b4")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(256)
                        .build())
                .layer(25, new BatchNormalization())
                .layer(26, new ActivationLayer.Builder().activation(Activation.RELU).build())
                .layer(27, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .padding(1, 1)
                        .name("maxpool1b4")
                        .build())
                //BLOCK 5
                .layer(28, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1})
                        .name("conv1b5")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(1024)
                        .build())
                .layer(29, new BatchNormalization())
                .layer(30, new ActivationLayer.Builder().activation(Activation.RELU).build())
                .layer(31, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1})
                        .name("conv2b5")
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(1024)
                        .build())
                .layer(32, new BatchNormalization())
                .layer(33, new ActivationLayer.Builder().activation(Activation.RELU).build())
                .layer(34, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .padding(1, 1)
                        .name("maxpool1b5")
                        .build())
                //CONDENSATION
                .layer(35, new DenseLayer.Builder()
                        .name("ffn1")
                        .nIn(50176)
                        .nOut(512)
                        .weightInit(new NormalDistribution(0, 0.005))
                        .biasInit(bias)
                        .dropOut(0.25)
                        .build())
                .layer(36, new DenseLayer.Builder()
                        .name("ffn2")
                        .nOut(64)
                        .weightInit(new NormalDistribution(0, 0.005))
                        .biasInit(bias)
                        .dropOut(0.25)
                        .build())
                .layer(37, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .weightInit(new NormalDistribution(0, 0.005))
                        .biasInit(0.1)
                        .build())
                .setInputType(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]))
                .build();
    }


    private NeuralNetConfiguration.ListBuilder convBlock(NeuralNetConfiguration.ListBuilder multiLayerConfiguratioListBuilder,
                                                         final int blockNumber,
                                                         final int filters) {
        return multiLayerConfiguratioListBuilder
                .layer(blockNumber * 7, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1})
                        .name("conv" + blockNumber + "b" + blockNumber)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Same)
                        .nIn(inputShape[0])
                        .nOut(filters)
                        .build())
                .layer(blockNumber * 7 + 1, new LocalResponseNormalization.Builder().build())
                .layer(blockNumber * 7 + 2, new ActivationLayer.Builder().activation(Activation.RELU).build())
                .layer(blockNumber * 7 + 3, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{2, 2})
                        .name("conv" + blockNumber + 1 + "b" + blockNumber)
                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                        .convolutionMode(ConvolutionMode.Same)
                        .nIn(inputShape[0])
                        .nOut(filters)
                        .build())
                .layer(blockNumber * 7 + 4, new LocalResponseNormalization.Builder().build())
                .layer(blockNumber * 7 + 5, new ActivationLayer.Builder().activation(Activation.RELU).build())
                .layer(blockNumber * 7 + 6, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .padding(1, 1)
                        .name("maxpool" + blockNumber + "b" + blockNumber)
                        .build());
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
                                .weightInit(new NormalDistribution(0, 0.2 * (2.0 / (4096 + numClasses)))) //This weight init dist gave better results than Xavier
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
