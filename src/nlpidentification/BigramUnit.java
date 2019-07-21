
package nlpidentification;

import java.util.Arrays;

/**
 * A class that represents the frequency at which a Bigram occurs.
 * @author DominicWild
 */
public class BigramUnit implements Comparable{

    private int freq;           //The frequnecy at which the bigram occurs.
    private char[] bigram;      //The bigram itself.

    /**
     * Basic constructor for a bigram.
     * @param c1 The first character of the bigram.
     * @param c2 The second character of the bigram.
     */
    public BigramUnit(char c1, char c2) {
        this.freq = 0;
        this.bigram = new char[]{c1, c2};
    }

    /**
     * Basic constructor for a bigram.
     * @param cArray An array of size 2, that is stored as a bigram.
     */
    public BigramUnit(char[] cArray) {
        this.freq = 0;
        if (cArray.length == 2) {
            this.bigram = cArray;
        } else {
            throw new IllegalArgumentException("Bigram had more than 2 characters");
        }
    }
    
    public String getBigram(){
        return new String(this.bigram);
    }
    
    public int getFreq(){
        return this.freq;
    }

    public void inc() {
        this.freq++;
    }
    
    public void inc(int amount) {
        this.freq += amount;
    }

    /**
     * Compares bigrams based on their frequency values. A bigram with a higher frequency, is classed as greater than one with a lower frequency.
     * @param o The Bigram to compare. Undefined for other objects.
     * @return Integer representing if the bigram is greater or not.
     */
    @Override
    public int compareTo(Object o) {
        if (o instanceof BigramUnit) {
            BigramUnit compare = (BigramUnit) o;
            return compare.getFreq() - this.freq;
        } else {
            throw new IllegalArgumentException("Bigrams may only be compared to bigrams");
        }
    }

    /**
     * Compares the equality of two bigrams. Done by comparing their bigram character values, they're equal if the order of these characters match.
     * @param obj The bigram to compare to. Undefined for other objects.
     * @return Boolean representing if the two BigramUnits are equal.
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        final BigramUnit other = (BigramUnit) obj;
        return Arrays.equals(this.bigram, other.bigram);
    }
    
    
}
