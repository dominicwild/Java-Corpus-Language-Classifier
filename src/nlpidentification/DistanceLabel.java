package nlpidentification;

/**
 * A container for a distance measure associated with a particular language.
 * Used to wrap the variables that predict what language a test text is.
 *
 * @author DominicWild
 */
public class DistanceLabel {

    private Language lang;              //The language this label represents.
    private int rankDistance;           //The distance from some comparison.

    public DistanceLabel(Language lang, int rankDistance) {
        this.lang = lang;
        this.rankDistance = rankDistance;
    }

    public Language getLang() {
        return lang;
    }

    public int getRankDistance() {
        return rankDistance;
    }


}
