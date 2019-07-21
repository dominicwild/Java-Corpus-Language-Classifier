package nlpidentification;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.StringTokenizer;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.lang3.StringEscapeUtils;

/**
 *
 * @author DominicWild
 */
public class DataSet {

    private final int CLEAN_BELOW = 2;              //Every bigram with a frequency below this value (exclusive) will be removed.
    protected ArrayList<BigramUnit> dataset;        //The dataset of all bigrams in this set.
    private Language dataSetLanguage;               //The language this data set has bigrams for.
    private String corpusFileName;                  //The name of the corpus file used to generate this data set.
    private int wordCount;                          //The amount of words this data set has processed.
    private int lineCount;                          //The amount of lines this data set has processed.

    /**
     * Creates a DataSet from the passed corpus file, labeled with the passed language.
     * @param corpusFile The corpus file to extract bigrams from.
     * @param lang The language the corpus file refers to.
     */
    public DataSet(String corpusFile, Language lang) {
        this.init(corpusFile, lang);
        this.populateTable();
        this.manageDataset();
    }

    /**
     * Creates a DataSet from the passed corpus file, labeled with the passed language. However, limiting itself to the processing only the passed word limit amount of words.
     * @param corpusFile The corpus file to extract bigrams from.
     * @param lang The language the corpus file refers to.
     * @param wordLimit The limit on the amount of words to process.
     */
    public DataSet(String corpusFile, Language lang, int wordLimit) {
        this.init(corpusFile, lang);
        this.populateTable(wordLimit);
        this.manageDataset();
    }
    
    /**
     * Creates a DataSet from the passed corpus file, labeled with the passed language. However, limiting itself to only the passed number of lines, starting from a particular startAt line.
     * @param corpusFile The corpus file to extract bigrams from.
     * @param lang The language the corpus file refers to.
     * @param startAt The line to start processing at.
     * @param lineLimit The amount of lines to process from startAt.
     */
    public DataSet(String corpusFile, Language lang, int startAt, int lineLimit) {
        this.init(corpusFile, lang);
        this.populateTableByLine(lineLimit,startAt);
        this.manageDataset();
    }
    
    /**
     * Initialises basic variables for the DataSet constructor.
     * @param corpusFile The corpus file to extract bigrams from.
     * @param lang The language the corpus file refers to.
     */
    private void init(String corpusFile, Language lang) {
        this.corpusFileName = corpusFile;
        this.wordCount = 0;
        this.lineCount = 0;
        this.dataset = new ArrayList<>();
        this.dataSetLanguage = lang;
    }

    /**
     * General post-processing that is done when data is gathered.
     */
    private void manageDataset() {
        this.clean();
        this.dataset.sort(null);
    }
    
    /**
     * Prints bigrams into a CSV file. With default name "corpusFileName" + Freq.csv.
     */
    public void printTable(){
        this.printTable(this.corpusFileName.substring(0, this.corpusFileName.indexOf(".")));
    }

    /**
     * Prints bigrams into a CSV file with the passed name.
     * @param fileName The name of the CSV file to create.
     */
    public void printTable(String fileName) {
        File output = new File(fileName + ".csv");

        try (FileWriter writer = new FileWriter(output)) {

            if (!output.exists()) {
                output.createNewFile();
            }
            
            for (BigramUnit unit : this.dataset) {
                writer.write(StringEscapeUtils.escapeCsv(unit.getBigram())+ "," + unit.getFreq() + System.lineSeparator());
            }

            writer.flush();
        } catch (IOException ex) {
            Logger.getLogger(DataSet.class.getName()).log(Level.SEVERE, "Error printing table to file.", ex);
        }
    }

    /**
     * Populates DataSet with bigrams from the passed file.
     * @param wordLimit The limit of words to process.
     */
    protected void populateTable(int wordLimit) {
        try (BufferedReader br = new BufferedReader(new FileReader(this.corpusFileName))) {

            HashMap<String, BigramUnit> freqMap = new HashMap<>(); //Mapping of bigrams to BigramUnits.
            String line = br.readLine();
            boolean stop = false;       //Determines when to stop processing.

            while (line != null) {
                stop = this.processLine(freqMap, line, wordLimit);
                line = br.readLine();
                if (stop) {
                    break;
                }
            }
            this.dataset.addAll(freqMap.values()); //Add all bigram units to the dataset.
        } catch (FileNotFoundException ex) {
            Logger.getLogger(DataSet.class.getName()).log(Level.SEVERE, "File not found for " + this.corpusFileName, ex);
        } catch (IOException ex) {
            Logger.getLogger(DataSet.class.getName()).log(Level.SEVERE, "IOException occured for " + this.corpusFileName, ex);
        }
    }
    
    /**
     * Populate DataSet limiting processing on a line basis.
     * @param lineLimit The limit of lines to process.
     */
    protected void populateTableByLine(int lineLimit){
        this.populateTableByLine(lineLimit, 0);
    }
    
    /**
     * Populate DataSet limiting processing on a line basis, starting at a specific line.
     * @param lineLimit The limit of lines to process.
     * @param startAt The line to start processing at.
     */
    protected void populateTableByLine(int lineLimit, int startAt){
        try (BufferedReader br = new BufferedReader(new FileReader(this.corpusFileName))) {

            HashMap<String, BigramUnit> freqMap = new HashMap<>();
            String line;
            int lineIndex = 0; 
            boolean stop = false;

            while(lineIndex < startAt){ //Run br.readLine() startAt times.
                lineIndex++;
                br.readLine();
            }
            
            while ((line = br.readLine()) != null && this.lineCount < lineLimit) {
                stop = this.processLine(freqMap, line);
                if (stop) {
                    break;
                }
            }
            
            this.dataset.addAll(freqMap.values()); //Add all bigram units to the dataset.
        } catch (FileNotFoundException ex) {
            Logger.getLogger(DataSet.class.getName()).log(Level.SEVERE, "File not found for " + this.corpusFileName, ex);
        } catch (IOException ex) {
            Logger.getLogger(DataSet.class.getName()).log(Level.SEVERE, "IOException occured for " + this.corpusFileName, ex);
        }
    }

    /**
     * Processes a line for bigrams.
     * @param freqMap The map to store the result of processing within.
     * @param line The line to process.
     * @return Whether or not to stop processing from this point.
     */
    protected boolean processLine(HashMap<String, BigramUnit> freqMap, String line){
       return this.processLine(freqMap, line, Integer.MAX_VALUE);
    }
    
    /**
     * Processes a line for bigrams.
     * @param freqMap The map to store the result of processing within.
     * @param line The line to process.
     * @param wordLimit The limit of words to process.
     * @return Whether or not to stop processing from this point.
     */
    protected boolean processLine(HashMap<String, BigramUnit> freqMap, String line, int wordLimit) {
        this.lineCount++;               
        char[] bigram = new char[2];
        boolean stop = false;

        StringTokenizer tokenizer = new StringTokenizer(line);
        wordCount += tokenizer.countTokens();
        if (wordLimit < wordCount) { //If we're going to go over the word limit
            line = ""; //We only want to limit our line to words we need to process.
            this.wordCount -= tokenizer.countTokens();
            int numWordsToGet = wordLimit - wordCount;
            this.wordCount += numWordsToGet;
            for (int i = 0; i < numWordsToGet; i++) {
                line += tokenizer.nextToken() + " ";
            }
            line = line.trim(); //Get rid of space at the end
            stop = true;
        }

        for (int i = 0; i < line.length() - 1; i++) { 
            char c = line.charAt(i);
            char cNext = line.charAt(i + 1);
            bigram[0] = c;
            bigram[1] = cNext;
            String strBigram = new String(bigram);
            if (freqMap.containsKey(strBigram)) { //Increment bigram if it exists, if not add it and increment.
                freqMap.get(strBigram).inc();
            } else {
                BigramUnit unit = new BigramUnit(bigram.clone());
                unit.inc();
                freqMap.put(strBigram, unit);
            }
        }

        return stop;
    }

    /**
     * Cleans the DataSet by defined means.
     */
    public void clean() {
        this.dataset.removeIf(p -> {
            return p.getFreq() < CLEAN_BELOW;
        });
    }

    public ArrayList<BigramUnit> getDataset() {
        return dataset;
    }

    public Language getDataSetLanguage() {
        return dataSetLanguage;
    }

    /**
     * Populates the DataSet with bigrams from the file name attached to the object.
     */
    protected void populateTable() {
        this.populateTable(Integer.MAX_VALUE);
    }

    public int getWordCount() {
        return wordCount;
    }

    public String getCorpusFileName() {
        return corpusFileName;
    }

    public void setWordCount(int wordCount) {
        if (wordCount >= 0) {
            this.wordCount = wordCount;
        } else {
            throw new IllegalArgumentException("Negative number provided.");
        }
    }

    public int getLineCount() {
        return lineCount;
    }
    
    /**
     * Creates an array of folded DataSets around a certain file.
     * @param fileName The name of the file to create folds of.
     * @param lang The language of the specified file.
     * @param numFolds The number of folds to include.
     * @return The set of folds.
     */
    public static DataSet[][] folds(String fileName,Language lang,int numFolds){
        
        DataSet[][] folds = new DataSet[numFolds][2];
        int factionOfLines = (int) ((1.0/numFolds)*linesInFile(fileName)); //Defines the step we take through the file
        
        for(int i=0;i<numFolds;i++){
            int bound1 = i*factionOfLines; //Determine both bounds for our fold
            int bound2 = (i+1)*factionOfLines;
            DataSet set1 = new DataSet(fileName,lang,0,bound1);
            DataSet set2 = new DataSet(fileName,lang,bound1,factionOfLines);
            DataSet set3 = new DataSet(fileName,lang,bound2,Integer.MAX_VALUE);
            folds[i][0] = DataSet.combine(set1, set3); //Combine sets outside the fold portion
            folds[i][1] = set2; //Set our fold portion
        }
        
        return folds;
    }
    
    /**
     * Combines two DataSets into one. Done by adding frequency values onto BigramUnit's and adding ones that aren't present yet.
     * @param set1 The first set to combine.
     * @param set2 The second set to combine.
     * @return The combined 2 sets into one DataSet.
     */
    public static DataSet combine(DataSet set1, DataSet set2){
        ArrayList<BigramUnit> data1 = set1.getDataset();
        ArrayList<BigramUnit> data2 = set2.getDataset();
        
        for(BigramUnit unit : data1){ //For each BigramUnit in data1 
            if(data2.contains(unit)){ //Increment frequency from same units by the data1 unit's
                data2.get(data2.indexOf(unit)).inc(unit.getFreq());
            } else { //Or add it, if it is not present.
                data2.add(unit);
            }
        }
        //Update semantics about the files
        set2.setLineCount(set2.getLineCount() + set1.getLineCount());
        set2.setWordCount(set2.getWordCount() + set1.getWordCount());
        return set2;
    }
    
    /**
     * Calculates the number of lines within a text file.
     * @param fileName The name of the file to find the number of lines within.
     * @return The number of lines in the file.
     */
    public static int linesInFile(String fileName){
        try {
            return Files.readAllLines(Paths.get(fileName)).size();
        } catch (IOException ex) {
            Logger.getLogger(DataSet.class.getName()).log(Level.SEVERE, null, ex);
        }
        return -1;
    }

    public void setLineCount(int lineCount) {
        this.lineCount = lineCount;
    }
    
    

}
