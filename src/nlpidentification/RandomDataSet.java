
package nlpidentification;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A dataset that randomises the data it gets from a file.
 * @author DominicWild
 */
public class RandomDataSet extends DataSet {

    /**
     * Defines a RandomDataSet from a specified file.
     * @param corpusFile The name of the file to get the data from.
     * @param lang The language label associated with the data.
     * @param wordLimit The limit on the amount of words from the file.
     */
    public RandomDataSet(String corpusFile, Language lang, int wordLimit) {
        super(corpusFile, lang, wordLimit);
    }
    
    /**
     * Defines a RandomDataSet from a specified file.
     * @param corpusFile The name of the file to get the data from.
     * @param lang The language label associated with the data.
     */
    public RandomDataSet(String corpusFile, Language lang) {
        super(corpusFile, lang, Integer.MAX_VALUE);
    }

    /**
     * Populates RandomDataSet with bigrams up to a specified amount of words from the file. However, gets the data randomly from the file on a line by line basis.
     * @param wordLimit The limit of words to process.
     */
    @Override
    protected void populateTable(int wordLimit) {
        HashMap<String, BigramUnit> freqMap = new HashMap<>();
        if (Integer.MAX_VALUE == wordLimit) { //If we're getting all words, nothing to randomize.
            super.populateTable(wordLimit);
        } else {
            try {
                List<String> lines = Files.readAllLines(Paths.get(this.getCorpusFileName()));
                int randIndex;      //Random line index
                String line;
                boolean stop = false;
                while (!stop && lines.size() > 0) {
                    randIndex = (int) (lines.size() * Math.random());
                    line = lines.get(randIndex);
                    lines.remove(randIndex); //Remove, so we can't pick the same line again.
                    stop = this.processLine(freqMap, line, wordLimit);
                }
                this.dataset.addAll(freqMap.values());
            } catch (IOException ex) {
                Logger.getLogger(RandomDataSet.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }

    /**
     * Generate a new random set of data with the specified word limit.
     * @param wordLimit The limit of words to process.
     */
    public void newRandomSet(int wordLimit) {
        if (wordLimit >= 0) {
            this.setWordCount(0); //Reset semantics
            this.setLineCount(0);
            this.dataset = new ArrayList<>();
            this.populateTable(wordLimit);
            this.clean();
            this.dataset.sort(null);
        } else {
            throw new IllegalArgumentException("Negative number not valid." + wordLimit);
        }
    }


}
