package utils;

import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Random;

public class DataLoader {

    private final static String TEST_PATH = "test";
    private final static String TRAIN_PATH = "train";

    private final static int SEED = 123;

    public static Map<String, FileSplit> loadDataFiles(){
        final Random randNumGen = new Random(SEED);

        final Map<String, FileSplit> data = new HashMap<>();

        final ClassLoader loader = DataLoader.class.getClassLoader();
        final File trainData = new File(Objects.requireNonNull(loader.getResource(TEST_PATH)).getFile());
        final File testData = new File(Objects.requireNonNull(loader.getResource(TRAIN_PATH)).getFile());

        final FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        final FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        data.put("train",train);
        data.put("test",test);

        return data;
    }
}
