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

    public static Map<String, FileSplit> loadDataFiles() {
        final Map<String, FileSplit> data = new HashMap<>();

        final FileSplit train = getFileSplit(TEST_PATH);
        final FileSplit test = getFileSplit(TRAIN_PATH);

        data.put("train", train);
        data.put("test", test);

        return data;
    }

    private static FileSplit getFileSplit(final String path) {
        final Random randNumGen = new Random(SEED);
        final ClassLoader loader = DataLoader.class.getClassLoader();
        final File file = new File(Objects.requireNonNull(loader.getResource(path)).getFile());
        return new FileSplit(file, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
    }
}
