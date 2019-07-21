package nlpidentification;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Objects;
import java.util.StringTokenizer;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author DominicWild
 */
public class NLPIdentification {
    
    private static final int TRIALS_HOME_MIN_SIZE = 1000;       //Number of repeated runs to verify minimum word size
    //Directory organization constants
    private static final String ROOT_DIR = "LanguageData";
    private static final String FREQ_DIR = ROOT_DIR + "/BigramFrequencyTables/";
    private static final String CROSS_VALIDATION_DIR = ROOT_DIR + "/CrossValidation/";
    private static final String HOME_MIN_TEST_DIR = ROOT_DIR + "/MiniumTestSample/";
    private static final String VAR_TRAINING_SIZE_DIR = ROOT_DIR + "/VariableTrainingSize/";
    private static PrintWriter results;                 //Writer for our results file.

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        NLPIdentification nlpIdentification = new NLPIdentification();
    }

    public NLPIdentification() {
        //Make necessary directories if they don't exist
        new File(ROOT_DIR).mkdir();
        new File(FREQ_DIR).mkdir();
        new File(CROSS_VALIDATION_DIR).mkdir();
        new File(HOME_MIN_TEST_DIR).mkdir();
        new File(VAR_TRAINING_SIZE_DIR).mkdir();
        //Execute our main program
        try {
            this.execute();
        } catch (FileNotFoundException ex) {
            Logger.getLogger(NLPIdentification.class.getName()).log(Level.SEVERE, "Error writing to results.", ex);
        }
        //Flush any potential remaining data and close print writer.
        results.flush();
        results.close();
    }

    public void execute() throws FileNotFoundException {
        //Remove tags from necessary files
        removeTags("englishTagged.txt", "english.txt");
        removeTags("czechTagged.txt", "czech.txt");
        //Split into training and test data.
        fileSplit(0.9, "english.txt", "englishTrain.txt", "englishTest.txt");
        fileSplit(0.9, "czech.txt", "czechTrain.txt", "czechTest.txt");
        fileSplit(0.9, "slovenian.txt", "slovenianTrain.txt", "slovenianTest.txt");
        fileSplit(0.9, "german.txt", "germanTrain.txt", "germanTest.txt");
        //Create dataset training objects used throughout testing.
        DataSet englishTrain = new DataSet("englishTrain.txt", Language.ENG);
        DataSet czechTrain = new DataSet("czechTrain.txt", Language.CZH);
        DataSet slovenianTrain = new DataSet("slovenianTrain.txt", Language.SLV);
        DataSet germanTrain = new DataSet("germanTrain.txt", Language.GER);
        //Print the bigram tables to disk for analysis
        englishTrain.printTable(FREQ_DIR + "englishFreq");
        czechTrain.printTable(FREQ_DIR + "czechFreq");
        slovenianTrain.printTable(FREQ_DIR + "slovenianFreq");
        germanTrain.printTable(FREQ_DIR + "germanFreq");
        //Make unique results file
        Date runTime = new Date(System.currentTimeMillis());
        results = new PrintWriter("Results" + new SimpleDateFormat("yyyy-MM-dd-SSS").format(runTime) + ".txt");
        //Initialize constants for experiments
        final int folds = 10;
        final int varTrainRuns = 100;
        int minWords = 0;
        double validatePercentage = 0;
        final double acceptPercentage = 0.95;
        logPrintln("--------------------------------Cross Validation--------------------------------");
        validatePercentage = Math.round(crossValidate(folds,"english.txt",Language.ENG,czechTrain, slovenianTrain, germanTrain)/folds*100.0);
        logPrintln("English Validation Pass Rate with " + folds + " folds: " + validatePercentage + "%");
        validatePercentage = Math.round(crossValidate(folds,"german.txt",Language.GER,czechTrain, slovenianTrain, englishTrain)/folds*100.0);
        logPrintln("German Validation Pass Rate with " + folds + " folds: " + validatePercentage + "%");
        validatePercentage = Math.round(crossValidate(folds,"slovenian.txt",Language.SLV,czechTrain, englishTrain, germanTrain)/folds*100.0);
        logPrintln("Slovenian Validation Pass Rate with " + folds + " folds: " + validatePercentage + "%");
        validatePercentage = Math.round(crossValidate(folds,"czech.txt",Language.CZH,englishTrain, slovenianTrain, germanTrain)/folds*100.0);
        logPrintln("Czech Validation Pass Rate with " + folds + " folds: " + validatePercentage + "%");
        logPrintln("--------------------------------Variable Sized Training Sets Fails--------------------------------");
        variableTrainRun("english.txt", new DataSet("englishTest.txt", Language.ENG), varTrainRuns, czechTrain, slovenianTrain, germanTrain);
        variableTrainRun("german.txt", new DataSet("germanTest.txt", Language.GER), varTrainRuns, czechTrain, slovenianTrain, englishTrain);
        variableTrainRun("slovenian.txt", new DataSet("slovenianTest.txt", Language.SLV), varTrainRuns, czechTrain, englishTrain, germanTrain);
        variableTrainRun("czech.txt", new DataSet("czechTest.txt", Language.CZH), varTrainRuns, englishTrain, slovenianTrain, germanTrain);
        logPrintln("--------------------------------Homing on Minimum Size Test That Can Be Predicted--------------------------------");
        minWords = calculateMinimumTestSample("englishTest.txt", Language.ENG, acceptPercentage, englishTrain, czechTrain, slovenianTrain, germanTrain);
        logPrintln("For English we can predict " + minWords + " words minimum with the training model that we have." );
        minWords = calculateMinimumTestSample("germanTest.txt", Language.GER, acceptPercentage, englishTrain, czechTrain, slovenianTrain, germanTrain);
        logPrintln("For German we can predict " + minWords + " words minimum with the training model that we have." );
        minWords = calculateMinimumTestSample("slovenianTest.txt", Language.SLV, acceptPercentage, englishTrain, czechTrain, slovenianTrain, germanTrain);
        logPrintln("For Slovenian we can predict " + minWords + " words minimum with the training model that we have." );
        minWords = calculateMinimumTestSample("czechTest.txt", Language.CZH, acceptPercentage, englishTrain, czechTrain, slovenianTrain, germanTrain);
        logPrintln("For Czech we can predict " + minWords + " words minimum with the training model that we have." );
    }
    
    
    /**
     * Cross validates a given corpus using a fold method. The results are logged within a csv.
     * @param folds The number of folds to use.
     * @param validateCorpus The corpus to run the validation on.
     * @param langValidate The language of the validation corpus.
     * @param otherTrainSets The other training sets to compare against.
     * @return The number of correctly identified folds.
     */
    public int crossValidate(int folds, String validateCorpus, Language langValidate, DataSet... otherTrainSets) {
        
        DataSet[][] validateSets = DataSet.folds(validateCorpus, langValidate, folds); //Gets an array of folds.
        int numCorrect = 0;

        try (PrintWriter writer = new PrintWriter(CROSS_VALIDATION_DIR +langValidate + "CrossValidate.csv")) {
            ArrayList<DataSet> trainingSets = new ArrayList<>();
            for (DataSet toAdd : otherTrainSets) { //Add all training sets to an array list to mutate through iteration.
                trainingSets.add(toAdd);
            }

            for (int i = 0; i < validateSets.length; i++) {
                trainingSets.add(validateSets[i][0]); //Add the training set at index 0
                Language predicted = this.predictSampleLanguage(validateSets[i][1], writer, trainingSets.toArray(new DataSet[trainingSets.size()]));
                trainingSets.remove(validateSets[i][0]); //Remove same set to test the next iteration
                if (predicted == langValidate) { //Log amount of correct predictions
                    numCorrect++;
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(NLPIdentification.class.getName()).log(Level.SEVERE, null, ex);
        }
        return numCorrect;
    }

    /**
     * Calculates the minimum amount of words that can be correctly identified within a minimum percentage of correctness.
     * @param sampleFile The corpus file to use for this test.
     * @param expectedLanguage The expected language of the corpus.
     * @param accuracyTarget The target accuracy we must maintain.
     * @param trainSets All other training sets to test against.
     * @return The minimum number of words we can correctly identify.
     */
    public int calculateMinimumTestSample(String sampleFile, Language expectedLanguage, double accuracyTarget, DataSet... trainSets) {

        RandomDataSet sampleSet = new RandomDataSet(sampleFile, expectedLanguage);
        PrintWriter writer = null;
        try {
            writer = new PrintWriter(HOME_MIN_TEST_DIR + expectedLanguage + "MinTestSample.csv");
        } catch (FileNotFoundException ex) {
            Logger.getLogger(NLPIdentification.class.getName()).log(Level.SEVERE, null, ex);
        }

        int currentLimit = sampleSet.getWordCount();
        int previousLimit = 0;

        while (currentLimit != previousLimit) { //If our limits are equal, we're repeating something, probably have hit the lowest value.
            int correct = 0;
            for (int i = 0; i < TRIALS_HOME_MIN_SIZE; i++) {
                sampleSet.newRandomSet(currentLimit); 
                Language result = this.predictSampleLanguage(sampleSet, writer, trainSets);
                if (Objects.equals(result, expectedLanguage)) {
                    correct++;
                }
            }
            double accuracy = (double) correct / TRIALS_HOME_MIN_SIZE;
            logPrintln("[" + expectedLanguage + "]"+ "With " + currentLimit + " words we get " + accuracy*100 + "%");
            if (accuracy >= accuracyTarget) { //If we maintain our target, run again with half size of last test sample.
                previousLimit = currentLimit;
                currentLimit /= 2;
            } else { //Otherwise, stop and break from the loop.
                break;
            }
        }
        writer.flush();
        writer.close();
        return previousLimit; //Return the limit that didn't fail
    }

    /**
     * Removes tags from a file. Therefore only maintaining the textual content between tags.
     * @param fileName The name of the file to remove tags from.
     * @param newFileName The name of the new to output this content to.
     */
    private void removeTags(String fileName, String newFileName) {

        try (BufferedReader reader = new BufferedReader(new FileReader(fileName));) {
            FileWriter writer = new FileWriter(newFileName);
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.replaceAll("\\<.*?>", "").trim(); //Match anything between tags and replace it with nothing.
                if (!line.isEmpty()) { //If the line isn't empty after remoing the tags
                    writer.write(line + System.lineSeparator());
                }
            }
            writer.flush();
        } catch (FileNotFoundException ex) {
            Logger.getLogger(NLPIdentification.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(NLPIdentification.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * Tests training sets of various sizes against a constant test sample and logs this into a CSV for analysis.
     * @param fileName  The name of the corpus to test on.
     * @param testSample The test sample to use on all the runs.
     * @param numRuns The number of times we should run this. This also determines the increments of how fast the test size grows.
     * @param otherTrainSets All the other training sets to test against.
     */
    public void variableTrainRun(String fileName, DataSet testSample, int numRuns, DataSet... otherTrainSets) {

        int wordCount = numberOfWordsInFile(fileName);
        int step = Math.floorDiv(wordCount, numRuns);
        Language langExpected = testSample.getDataSetLanguage();
        ArrayList<DataSet> trainSets = new ArrayList<>();
        ArrayList<DistanceLabel> distanceMetrics = new ArrayList<>();
        try (PrintWriter writer = new PrintWriter(VAR_TRAINING_SIZE_DIR + langExpected + "VariableSizeTrainRuns.csv")) {
            
            for (DataSet trainSet : otherTrainSets) { //Compute distance labels for all sets we're only need to test once. We do this so that we don't need to recompute them on every iteration.
                trainSets.add(trainSet);
                DistanceLabel label = new DistanceLabel(trainSet.getDataSetLanguage(), determineDiffValue(trainSet, testSample));
                distanceMetrics.add(label);
                csvLog(trainSet, testSample, label, writer);
            }

            for (int i = step; i < step * numRuns; i += step) { //i represents the number of words we take per iteration.
                DataSet trainSet = new RandomDataSet(fileName, langExpected, i);
                DistanceLabel label = new DistanceLabel(trainSet.getDataSetLanguage(), determineDiffValue(trainSet, testSample));
                csvLog(trainSet, testSample, label, writer);
                distanceMetrics.add(label); //Add the label, use it for prediction, then remove it for the next iteration.
                Language predicted = this.classifyLanguage(distanceMetrics);
                distanceMetrics.remove(label);
                if (predicted != langExpected) {
                    logPrintln(langExpected + " model set with " + trainSet.getWordCount() + " words failed predicting " + predicted + " instead.");
                }
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(NLPIdentification.class.getName()).log(Level.SEVERE, null, ex);
        }

    }

    /**
     * Log a test we have conducted and the rank distance we got from our result with the specified PrintWriter.
     * @param trainSet The train set used.
     * @param testSet The test set used.
     * @param label The resulting DistanceLabel, containing the distance metric to store.
     * @param writer The writer to the CSV file we want to store this result in.
     */
    public void csvLog(DataSet trainSet, DataSet testSet, DistanceLabel label, PrintWriter writer) {
        writer.write(testSet.getWordCount() + "," + testSet.getDataSetLanguage() + ","
                + label.getRankDistance() + ","
                + trainSet.getDataSetLanguage() + "," + trainSet.getWordCount()
                + "\n");
    }

    /**
     * Predicts language of a passed sample and returns the resultant predicted language. All while logging the result within the sent PrintWriter.
     * @param testSample The test sample we want to predict.
     * @param writer The writer we want to output the log to.
     * @param training The training sets we wish to use to classify the test sample.
     * @return The predicted language.
     */
    public Language predictSampleLanguage(DataSet testSample, PrintWriter writer, DataSet... training) {
        ArrayList<DistanceLabel> distanceMetrics = new ArrayList<>();
        for (DataSet trainSet : training) {
            DistanceLabel label = new DistanceLabel(trainSet.getDataSetLanguage(), this.determineDiffValue(trainSet, testSample));
            distanceMetrics.add(label);
            //TestSize,TestLabel,Distance,TrainLabel,TrainSize in CSV
            if (writer != null) {
                csvLog(trainSet,testSample,label,writer);
            }
        }
        return classifyLanguage(distanceMetrics);
    }

     /**
     * Predicts language of a passed sample and returns the resultant predicted language.
     * @param testSample The test sample we want to predict.
     * @param training The training sets we wish to use to classify the test sample.
     * @return The predicted language.
     */
    public Language predictSampleLanguage(DataSet testSample, DataSet... training) {
        return predictSampleLanguage(testSample, null, training);
    }

    /**
     * Given a set of labels, picks the most likely candidate to classify the target language.
     * @param labels The labels that have pre-compute distance values.
     * @return The language we classify.
     */
    public Language classifyLanguage(ArrayList<DistanceLabel> labels) {
        Language predictedLanguage = null;
        int lowestDistance = Integer.MAX_VALUE;     //Allows us to get a guranteed match on first comparison
        int secondLowest = lowestDistance;
        DistanceLabel lowest = null;

        for (DistanceLabel label : labels) {
            if (lowestDistance > label.getRankDistance()) { //Find the label with the lowest distance
                lowest = label;
                lowestDistance = lowest.getRankDistance();
                predictedLanguage = label.getLang();
            }
        }
        labels.remove(lowest); //Remove this value
        for (DistanceLabel label : labels) { //Find the second lowest
            if (secondLowest > label.getRankDistance()) {
                secondLowest = label.getRankDistance();
            }
        }

        if (lowestDistance == secondLowest) { //If both are the same, we can't confidently say what language this is.
            return null;
        }
        return predictedLanguage;
    }

    /**
     * Calculates the rank-order difference value between a training set and a test set.
     * @param trainSet The training set to use in the distance value calculation.
     * @param testSet The test set to use in the distance value calculation.
     * @return The distance value between these sets.
     */
    public int determineDiffValue(DataSet trainSet, DataSet testSet) {
        int diff = 0;
        int listSize = testSet.getDataset().size();
        if (listSize > trainSet.getDataset().size()) { //If the test set, is larger than the training set
            listSize = trainSet.getDataset().size(); //Compare now only the training set number of elements
            diff += 1000 * (testSet.getDataset().size() - listSize); //Add the difference of things we couldn't compare to, to make a fair comparison.
        }

        //Get our limit of bigram values.
        ArrayList<BigramUnit> train = new ArrayList<>(trainSet.getDataset().subList(0, listSize));
        ArrayList<BigramUnit> test = new ArrayList<>(testSet.getDataset().subList(0, listSize));

        int index = 0;
        for (int i = 0; i < listSize; i++) {
            BigramUnit unit = test.get(i);
            index = train.indexOf(unit);
            if (index == -1) { //If invalid index, add default 1000
                diff += 1000;
            } else { //If valid, add the absolute difference
                diff += Math.abs(i - index);
            }
        }
        return diff;
    }

    

    /**
     * Split a file into two parts based on line number. The first file newFileName specified, will take the passed percentage amount of lines from the initial file specified to split.
     * @param percentage The percentage split, to the split the file with.
     * @param fileName The name of the file to split.
     * @param newFileName1 The name of the file to take the passed percentage amount of lines from the file to split.
     * @param newFileName2 The name of the file to take the remainder of the files contents from the split.
     */
    public void fileSplit(double percentage, String fileName, String newFileName1, String newFileName2) {
        if(percentage > 1 || percentage < 0){
            throw new IllegalArgumentException("Invalid percentage.");
        }
        try {
            int numLines = DataSet.linesInFile(fileName);
            File source = new File(fileName);
            File file1 = new File(newFileName1);
            File file2 = new File(newFileName2);
            file1.createNewFile();
            file2.createNewFile();

            int splitLines = (int) Math.floor(numLines * percentage); //Find the line at which we split the files.

            BufferedReader input = new BufferedReader(new FileReader(source));
            PrintWriter writer1 = new PrintWriter(file1); //Writer 1 takes first portion
            PrintWriter writer2 = new PrintWriter(file2);

            String line = input.readLine();
            int writer1Count = 0;
            int writer2Count = 0;

            while (line != null) {
                if (writer1Count > splitLines) { //If write 1 has its max amout of lines
                    writer2.write(line + "\n");
                    line = input.readLine();
                    continue;
                } else if (writer2Count > numLines - splitLines) {//If write 2 has its max amount of lines
                    writer1.write(line + "\n");
                    line = input.readLine();
                    continue;
                }
                if (Math.random() > 0.5) { //Add random line to either file.
                    writer2.write(line + "\n");
                    writer2Count++;
                } else {
                    writer1.write(line + "\n");
                    writer1Count++;
                }
                line = input.readLine();
            }

            writer2.flush();
            writer1.flush();
            writer2.close();
            writer1.close();

        } catch (IOException ex) {
            Logger.getLogger(NLPIdentification.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * Calculates the number of words within a file.
     * @param fileName The file to count the words in.
     * @return The number of words in the file.
     */
    public int numberOfWordsInFile(String fileName) {
        int count = 0;

        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            String line = reader.readLine();
            while (line != null) {
                count += new StringTokenizer(line).countTokens(); //Use string tokenizer to count the words
                line = reader.readLine();
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(NLPIdentification.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(NLPIdentification.class.getName()).log(Level.SEVERE, null, ex);
        }

        return count;
    }
    
    /**
     * Print both a log, to the results file and to the console.
     * @param output The string to output.
     */
    public void logPrintln(String output){
        results.println(output);
        System.out.println(output);
    }
    
}
