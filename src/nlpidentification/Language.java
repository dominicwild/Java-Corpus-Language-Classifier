package nlpidentification;

import java.nio.charset.Charset;

/**
 * Enum to represent a language.
 *
 * @author DominicWild
 */
public enum Language {
    ENG, CZH, SLV,GER;

    /**
     * String translations of all enumerations.
     * @return The string associated with a particular enumeration.
     */
    @Override
    public String toString() {
        switch (this) {
            case ENG:
                return "English";
            case CZH:
                return "Czech";
            case SLV:
                return "Slovenian";
            case GER:
                return "German";
            default:
                return "Unknown";
        }
    }
}
