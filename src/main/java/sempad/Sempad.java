package sempad;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import sempad.utils.PatternMatchIndices;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import sempad.utils.Util;

import java.io.*;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static sempad.utils.Util.*;

public class Sempad {
    private static String data_name;
    private static String inputFolder;
    private static String outputFolder;
    private static int noOfLines;
    private static int minimumSupport;
    private static int maxLength;
    private static StanfordCoreNLP pipeline;
    private static boolean load_sentenceBreakdownData;
    private static boolean load_metapatternData;
    private static int noOfPushUps;
    private static boolean includeContext;
    private static int startBatch;
    private static int batchSize;
    private static int splitSize;

    private static String[] nerTypes;
    private static List<String> stopWords;
    private static int noOfBatches = 0;
    private static Map<String, String> entityDictionary = new HashMap<>();
    private static Map<MetaPattern, Integer> singlePattern = new HashMap<>();
    private static Map<MetaPattern, Integer> allPattern = new HashMap<>();
    private static Map<MetaPattern, Integer> multiPattern = new HashMap<>();
    private static ForkJoinPool forkJoinPool = null;

    private static Logger logger = Logger.getLogger(Sempad.class.getName()) ;


    public Sempad()  {
    }

    /*********************************************** Initialization ***********************************************/

    private static void initialization() throws Exception {
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,parse,depparse");
        props.setProperty("tokenize.options", "splitHyphenated=false");
        pipeline = new StanfordCoreNLP(props);

        InputStream input = new FileInputStream("config.properties");
        Properties prop = new Properties();
        prop.load(input);
        if (!prop.containsKey("data_name") || prop.getProperty("data_name").equals("")) {
            throw new IOException("Parameter 'data_name' <does not exist/ is empty> in config.properties");
        }
        setOptionalParameterDefaults(prop);
        data_name = prop.getProperty("data_name");
        inputFolder = folderNameConsistency(prop.getProperty("inputFolder"));
        outputFolder = folderNameConsistency(prop.getProperty("outputFolder"));
        noOfLines = Integer.parseInt(prop.getProperty("noOfLines"));
        minimumSupport = Integer.parseInt(prop.getProperty("minimumSupport"));
        maxLength = Integer.parseInt(prop.getProperty("maxLength"));
        nerTypes = readList(new FileReader(prop.getProperty("nerTypes"))).toArray(new String[0]);
        stopWords = readList(new FileReader(prop.getProperty("stopWordsFile")));
        load_sentenceBreakdownData = Boolean.parseBoolean(prop.getProperty("load_sentenceBreakdownData"));
        load_metapatternData = Boolean.parseBoolean(prop.getProperty("load_metapatternData"));
        noOfPushUps = Integer.parseInt(prop.getProperty("noOfPushUps"));
        includeContext = Boolean.parseBoolean(prop.getProperty("includeContext"));
        startBatch = Integer.parseInt(prop.getProperty("startFromBatch"));
        batchSize = Integer.parseInt(prop.getProperty("batchSize"));
        splitSize = Integer.parseInt(prop.getProperty("splitSize"));
        int threadCount = Integer.parseInt(prop.getProperty("threadCount"));

        if (!new File(inputFolder).exists()) throw new IOException("Input Folder not found.");
        boolean metadata_exists = new File(inputFolder + "dict.json").exists();
        boolean annotatedDataExists = new File(inputFolder + "annotated.txt").exists();
        if (!metadata_exists || !annotatedDataExists) {
            throw new IOException("Required Input Files not found.");
        }
        if (load_sentenceBreakdownData) {
            String directory = inputFolder + "ProcessedInput/" + batchSize + "/";
            if (!new File(directory).exists()) throw new IOException("Sentence breakdown data does not exist for given batch size.");
        }
        makingDir(outputFolder);
        if (load_metapatternData) {
            String suffix = "." + noOfLines + "_" + minimumSupport + ".txt";
            boolean patternDataExists = new File(outputFolder + "allPatterns" + suffix).exists();
            if (!patternDataExists) {
                throw new IOException("Required Pattern Files not found.");
            }
        }
        makingDir(inputFolder + "ProcessedInput");
        makingDir(inputFolder + "ProcessedInput/" + batchSize);
        String logFileSuffix = "logFile." + noOfLines + "_" + minimumSupport;
        FileHandler logFile = new FileHandler(outputFolder + logFileSuffix + ".txt");
        logFile.setFormatter(new SimpleFormatter());
        logger.addHandler(logFile);
        try {
            forkJoinPool = new ForkJoinPool(threadCount);
        } catch (Exception e) {
            throw new Exception(e);
        }
    }

    /*********************************************** Build Dictionary, Build Sentences ***********************************************/
    private static void buildDictionary() throws Exception {
        logger.log(Level.INFO, "STARTING: Dictionary Building");
        ObjectMapper mapper = new ObjectMapper();
        entityDictionary = mapper.readValue(new File(
                inputFolder + "dict.json"), new TypeReference<Map<String, String>>() {});
        logger.log(Level.INFO, "COMPLETED: Dictionary Building");
    }

    // save each sentence to sentenceCollector, an iterable for meta pattern mining later
    private static void saveSentenceBreakdown(int batchIterNo, List<SentenceProcessor> sentenceCollector, int tempIter) throws Exception {
        logger.log(Level.INFO, "STARTING: Sentence Breakdown Serialization for batch " + batchIterNo);
        String directory = inputFolder + "ProcessedInput/" + batchSize + "/";
        String prefix = tempIter == -1 ? "sentenceBatch"  + batchIterNo + "." + noOfLines : "t" + batchIterNo;
        File file = new File(directory + prefix + (tempIter == -1 ? "" : tempIter) + ".txt");
        FileOutputStream fileOut = new FileOutputStream(file);
        ObjectOutputStream out = new ObjectOutputStream(fileOut);
        if (tempIter != -1) file.deleteOnExit();
        if (tempIter > 2) {
            File file1 = new File(directory + prefix  + (tempIter - 1) + ".txt");
            if (file1.delete()) logger.log(Level.INFO, "Deleted previous iteration file of batch " + batchIterNo);
            else logger.log(Level.INFO, "Could not delete previous iteration file of batch " + batchIterNo);
        }
        out.writeObject(sentenceCollector);
        out.close();
        fileOut.close();
        logger.log(Level.INFO, "COMPLETED: Sentence Breakdown Serialization for batch " + batchIterNo);
    }

    // parallel
    private static void processParallelHelper(List<SentenceProcessor> sentenceCollector, int batchIterNo) throws Exception {
        IntStream istream = IntStream.range(0, sentenceCollector.size());
        forkJoinPool.submit(() -> istream.parallel().forEach(s -> sentenceCollector.get(s).processSentence(pipeline, entityDictionary, nerTypes))
        ).get();
        istream.close();
        logger.log(Level.INFO, "PROCESSED: Batch Iteration Number - " + batchIterNo);
        saveSentenceBreakdown(batchIterNo, sentenceCollector, -1);
    }

    // parallel
    private static void buildSentences() throws Exception {
        logger.log(Level.INFO, "STARTING: Sentences Processing");
        String inputFile = inputFolder + "annotated.txt";
        int totalNoOfSentences = totalLines(inputFile, noOfLines);
        logger.log(Level.INFO, "Total Number of Sentences: " + totalNoOfSentences);

        List<SentenceProcessor> sentenceCollector = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(inputFile));
        String line = reader.readLine();

        int lineNo = 0;
        final boolean indexGiven = (line != null && line.split("\t").length != 1);
        int phraseCount = 0;
        Pattern pattern = Pattern.compile("[\\w\\d]+_[\\w\\d]+");
        while (line != null) {
            if (noOfLines > 0 && lineNo == noOfLines) {
                break;
            }
            if (!indexGiven || line.split("\t").length == 2) {
                String index = indexGiven ? line.split("\t")[0] : String.valueOf(lineNo);
                String sent = indexGiven ? line.split("\t")[1] : line;
                if (sent.split(" ").length > 4 && sent.split(" ").length < 100) {
                    if ( noOfBatches >= startBatch) {
                        Matcher matcher = pattern.matcher(sent);
                        Set<String> foundMatches = new HashSet<>();
                        while (matcher.find()) {
                            String match = matcher.group();
                            if (!foundMatches.contains(match)) {
                                foundMatches.add(match);
                                String newEntity = "PHRASEGEN" + phraseCount++;
                                sent = sent.replace(match, newEntity);
                                entityDictionary.put(newEntity, match);
                            }
                        }
                    }
                    sentenceCollector.add(new SentenceProcessor(sent, index));
                }
                if (sentenceCollector.size() == batchSize) {
                    if (noOfBatches >= startBatch) processParallelHelper(sentenceCollector, noOfBatches);
                    sentenceCollector.clear();
                    noOfBatches++;
                }
                lineNo++;
            }
            line = reader.readLine();
        }
        reader.close();

        if (sentenceCollector.size() != 0) {
            processParallelHelper(sentenceCollector, noOfBatches++);
            sentenceCollector.clear();
        }
        logger.log(Level.INFO, "COMPLETED: Sentences Processing");
    }

    /*********************************************** Frequent Pattern Mining ***********************************************/

    private static void clearPatterns() {
        singlePattern.clear();
        allPattern.clear();
        multiPattern.clear();
    }

    private static List<SentenceProcessor> loadSentenceBreakdown(int batchIterNo, int tempIter) throws Exception {
        logger.log(Level.INFO, "STARTING: Sentence Breakdown Deserialization for batch " + batchIterNo);
        String directory = inputFolder + "ProcessedInput/" + batchSize + "/";
        String prefix = tempIter == -1 ? "sentenceBatch" + batchIterNo + "." + noOfLines : "t" + batchIterNo + tempIter;
        FileInputStream fileIn = new FileInputStream( directory + prefix + ".txt");
        ObjectInputStream in = new ObjectInputStream(fileIn);
        List<SentenceProcessor> sentenceCollector = (List<SentenceProcessor>) in.readObject();
        in.close();
        fileIn.close();
        logger.log(Level.INFO, "COMPLETED: Sentence Breakdown Deserialization for batch " + batchIterNo);
        return sentenceCollector;
    }

    private static void classifyAndAddPattern(PatternCandidate patternCandidate) {
        MetaPattern metaPattern = patternCandidate.getMetaPattern();
        // Criteria 1: pattern informativeness, each pattern should either contain one entity mention plus at least one non-stop-word, or contain two or more entity mentions
        int nerCount = metaPattern.getNerCount();
        if (nerCount == 1) {
            singlePattern.put(metaPattern, singlePattern.getOrDefault(metaPattern, 0) + patternCandidate.getFrequency());
        } else if (nerCount > 1) {
            multiPattern.put(metaPattern, multiPattern.getOrDefault(metaPattern, 0) + patternCandidate.getFrequency());
        }
        if (nerCount >= 1) {
            allPattern.put(metaPattern, allPattern.getOrDefault(metaPattern, 0) + patternCandidate.getFrequency());
        }
    }

    private static void frequent_pattern_mining(int iteration, int start, int end) throws Exception {
        logger.log(Level.INFO, "STARTING: Frequent Pattern Mining - Iteration " + iteration);
        clearPatterns();
        //    private static void clearPatterns() {
        //        singlePattern.clear();
        //        allPattern.clear();
        //        multiPattern.clear();
        //    }

        List<SubSentWord> tokens = new ArrayList<>();
        Map<String, List<Integer>> dict_token = new HashMap<>();
        Map<String, List<Integer>> valid_pattern = new HashMap<>();
        List<SentenceProcessor> sentenceCollector;
        int tokenNumber = 0;

        for (int batchNo = start; batchNo < end; batchNo++) {
            // batchNo=0, start=0, end=1
            sentenceCollector = loadSentenceBreakdown(batchNo, iteration == 1 ? -1 : iteration);
            // sentenceCollector=ArrayList(size=60)

            for (SentenceProcessor sentence : sentenceCollector) {
                // sentence: 1233445_0	LOCATION235448 plays LOCATION326961 and LOCATION406449 meets LOCATION111212 on Friday.
                // (1) sentence(\t[1], e.g. "LOCATION235448 plays LOCATION326961 and LOCATION406449 meets LOCATION111212 on Friday."),
                // (2) sentID(\t[0], e.g. "1233445_0"),
                // (3) sentenceBreakdown(size=7, LinkedHasMap: SubSentWord->ArrayList,
                // e.g. verb will have its corresponding short sentence; nouns just itself; index is the position in sentence;
                // sentenceBreakdown[0](size=1): key={word="LOCATION235448", lemma="LOCATION", encoding="AA_nsubj", index=1}    (SubsentWord)
                //                       value=["0": {word="LOCATION235448", lemma="LOCATION", encoding="AA_nsubj", index=1}]    (ArrayList)

                // sentenceBreakdown[1](size=4): key={word="plays", lemma="play", encoding="A_root", index=2}
                //                       value=["0": {word="LOCATION235448", lemma="LOCATION", encoding="AA_nsubj", index=1},
                //                              "1": {word="plays", lemma="play", encoding="A_root", index=2},
                //                              "2": {word="LOCATION326961", lemma="LOCATION", encoding="AB_obj", index=3},
                //                              "3": {word=".", lemma=".", encoding="AD_punct", index=10}, ]

                // sentenceBreakdown[2](size=1): key={word="LOCATION326961", lemma="LOCATION", encoding="AB_obj", index=3}
                //                       value=["0": {word="LOCATION326961", lemma="LOCATION", encoding="AB_obj", index=3}]

                // sentenceBreakdown[3](size=1): key={word="LOCATION406449", lemma="LOCATION", encoding="ACB_nsubj", index=5}
                //                       value=["0": {word="LOCATION406449", lemma="LOCATION", encoding="ACB_nsubj", index=5}]

                // sentenceBreakdown[4](size=5): key={word="meets", lemma="meet", encoding="AC_conj", index=6}
                //                       value=["0": {word="LOCATION406449", lemma="LOCATION", encoding="ACB_nsubj", index=5},
                //                              "1": {word="meets", lemma="meet", encoding="AC_conj", index=6},
                //                              "2": {word="LOCATION111212", lemma="LOCATION", encoding="ACC_obj", index=7},
                //                              "3": {word="on", lemma="on", encoding="ACDA_case", index=8},
                //                              "4": {word="Friday", lemma="friday", encoding="ACD_obl", index=9}]

                // sentenceBreakdown[5](size=1): key={word="LOCATION111212", lemma="LOCATION", encoding="ACC_obj", index=7}
                //                       value=["0": {word="LOCATION111212", lemma="LOCATION", encoding="ACC_obj", index=7}]

                // sentenceBreakdown[6](size=1): key={word="Friday", lemma="friday", encoding="ACD_obl", index=9}
                //                       value=["0": {word="Friday", lemma="friday", encoding="ACD_obl", index=9}]
                // ),
                // (4) reverseEncoding(size=10, HashMap), // seems give each word in the sentence an encoding(including those not exist in sentenceBreakdown)
                // e.g.
                // "AB_obj" -> key="AB_obj" value={word="LOCATION326961", lemma="LOCATION", encoding="AB_obj", index=3}
                // "ACB_nsubj" -> key="ACB_nsubj" value={word="LOCATION406449", lemma="LOCATION", encoding="ACB_nsubj", index=5}
                // "AC_conj" -> key="AC_conj" value={word="meets", lemma="meet", encoding="AC_conj", index=6}
                // "AA_nsubj" -> key="AA_nsubj" value={word="LOCATION235448", lemma="LOCATION", encoding="AA_nsubj", index=1}
                // "ACD_obl" -> key="ACD_obl" value={word="Friday", lemma="friday", encoding="ACD_obl", index=9}
                // "ACA_cc" -> key="ACA_cc" value={word="and", lemma="and", encoding="ACA_cc", index=4}
                // "ACDA_case" -> key="ACDA_case" value={word="on", lemma="on", encoding="ACDA_case", index=8}
                // "A_root" -> key="A_root" value={word="plays", lemma="play", encoding="A_root", index=2}
                // "ACC_obj" -> key="ACC_obj" value={word="LOCATION111212", lemma="LOCATION", encoding="ACC_obj", index=7}
                // "AD_punct" -> key="AD_punct" value={word=".", lemma=".", encoding="AD_punct", index=10}
                // (5) replaceSurfaceName(size=0, HashMap),
                // (6) pushedUpSentences(size=0, HashMap)

                // senBreak is (3)sentenceBreakdown in sentence
                Map<SubSentWord, List<SubSentWord>> sentBreak = sentence.getSentenceBreakdown();
                // sentBreak.keySet() size = 7 the key of each (3)sentenceBreakdown[i]
                // 0: {word="LOCATION235448", lemma="LOCATION", encoding="AA_nsubj", index=1}
                // 1: {word="plays", lemma="play", encoding="A_root", index=2}
                // 2: {word="LOCATION326961", lemma="LOCATION", encoding="AB_obj", index=3}
                // 3: {word="LOCATION406449", lemma="LOCATION", encoding="ACB_nsubj", index=5}
                // 4: {word="meets", lemma="meet", encoding="AC_conj", index=6}
                // 5: {word="LOCATION111212", lemma="LOCATION", encoding="ACC_obj", index=7}
                // 6: {word="Friday", lemma="friday", encoding="ACD_obl", index=9}

                // Criteria 3: pattern completeness in semantics, we require that the pattern should appear at least once as a short sentence.
                for (SubSentWord key : sentBreak.keySet()) {
                    // key = {word="LOCATION235448", lemma="LOCATION", encoding="AA_nsubj", index=1}
                    // key = {word="plays", lemma="play", encoding="A_root", index=2}
                    // key = {word="LOCATION326961", lemma="LOCATION", encoding="AB_obj", index=3}
                    List<SubSentWord> list = sentBreak.get(key);
                    // list = ["0": {word="LOCATION235448", lemma="LOCATION", encoding="AA_nsubj", index=1}]
                    // list: value of each sentenceBreakdown[i]

                    for (SubSentWord word : list) {
                        // word = {word="LOCATION235448", lemma="LOCATION", encoding="AA_nsubj", index=1}
                        tokens.add(word);
                        // tokens (ArrayList) every value of each sentenceBreakdown[i]
                        // [0={SubSentWord}={word="LOCATION235448", lemma="LOCATION", encoding="AA_nsubj", index=1},
                        //  1={SubSentWord}={word="$", lemma="$", encoding="", index=-1},
                        //  2={SubSentWord}={word="LOCATION235448", lemma="LOCATION", encoding="AA_nsubj", index=1},
                        //  3={SubSentWord}={word="plays", lemma="play", encoding="A_root", index=2},
                        //  4={SubSentWord}={word="LOCATION326961", lemma="LOCATION", encoding="AB_obj", index=3},
                        //  5={SubSentWord}={word=".", lemma=".", encoding="AD_punct", index=10}]
                        //  6={SubSentWord}={word="$", lemma="$", encoding="", index=-1},
                        //  7={SubSentWord}={word="LOCATION326961", lemma="LOCATION", encoding="AB_obj", index=3},
                        //  8={SubSentWord}={word="$", lemma="$", encoding="", index=-1}]
                        updateMapCount(dict_token, word.getLemma(), tokenNumber++);
                        // dict_token (HashMap)
                        // {"LOCATION"->ArrayList [0, 2, 4, 7],
                        //  "play"->ArrayList [3],
                        //  "."->ArrayList [5]}
                        // those does not in the list [0,2,4,7]+[3]+[5] (that is, 1,6) is $ (separator of each SubSentWord)
                        updateMapCount(valid_pattern, word.getLemma(), 1);
                        // valid_pattern (HashMap)
                        // {"LOCATION"->ArrayList [1, 1, 1, 1],
                        //  "play"->ArrayList [1],
                        //  "."->ArrayList [1]}
                    }
                    tokens.add(new SubSentWord("$", "$", "", -1));
                    tokenNumber++;
                }
                for (SubSentWord key : sentence.getPushedUpSentences().keySet()) {
                    for (SubSentWord word : sentence.getPushedUpSentences().get(key)) {
                        tokens.add(word);
                        updateMapCount(dict_token, word.getLemma(), tokenNumber++);
                        updateMapCount(valid_pattern, word.getLemma(), 1);
                    }
                    tokens.add(new SubSentWord("$", "$", "", -1));
                    tokenNumber++;
                }
            }
            sentenceCollector.clear();
        }

        // Noisy Pattern Filtering
        // iterate over dict_token
        int patternLength = 1;
        while(dict_token.size() > 1) {
            // dict_toke size=691
            // maxLength=15
            if (patternLength > maxLength) break;
            patternLength++;
            // patternLength=2

            Map<String, List<Integer>> newdict_token = new HashMap<>();
            Map<String, List<Integer>> newvalid_pattern = new HashMap<>();

            for (String pattern: dict_token.keySet()){
                // pattern: "prepare", "sweetheart", "year", "hall", ("LOCATION")
                // pattern: "ORGANIZATION who", "the ORGANIZATION"
                List<Integer> positions = dict_token.get(pattern);
                // positions: [1092], [3730, 3739], [1390, 1401, 2966, 2995], [1759, 1767], ([0,2,4,7,9,11,...])
                // positions: [2716], [488, 1080, 2451, 2601, 2668],
                List<Integer> valid = valid_pattern.get(pattern);
                // valid: [1], [1,1], [1,1,1,1], [1,1]
                int frequency = positions.size();

                // minimumSupport=2
                if (frequency >= minimumSupport){

                    if (patternLength > 2 && sum(valid) >= minimumSupport){
                        PatternCandidate patternCandidate = new PatternCandidate(pattern, nerTypes, stopWords, frequency);
                        if (patternCandidate.isMetaPattern()) {
                            // Update AllPattern
                            classifyAndAddPattern(patternCandidate);
                        }
                    }

                    for (Integer i : positions){
                        if (i + 1 < tokenNumber){
                            SubSentWord nextToken = tokens.get(i + 1);
                            // "sweatheart": "3730: $", "3739: $",
                            // "year": "1390: apart", "1401: $", "2966: ,", "2995: $",
                            // "hall": "1759: .", "1767: for"
                            if (nextToken.getLemma().equals("$")) continue;
                            String newPattern = pattern + " " + nextToken.getLemma();
                            // "year": "1390: year apart", "2966: year ,"
                            // "hall": "1759: hall .", "1767: hall for"
                            int new_valid = 1;

                            List<String> pattern_tree = new ArrayList<>();
                            for (int j = i - patternLength + 2; j < i + 2; j++){
                                // getTrimmedEncoding() {return this.encoding.split("_")[0];}
                                pattern_tree.add(tokens.get(j).getTrimmedEncoding());
                                // "year", patternLength=2
                                // i = 1390
                                // tokens.get(j) = {word:"years", lemma="year", encoding="ACEA_obl:npmod", index=19},
                                //                 {word:"apart", lemma="apart", encoding="ACE_advmod", index=20}
                                // tokens.get(j).getTrimmedEncoding()="ACEA", "ACE"
                                // pattern_tree = ["ACEA", "ACE"]
                                // i = 2966
                                // tokens.get(j) = {word:"years", lemma="year", encoding="AACDD_obl", index=17},
                                //                 {word:",", lemma=",", encoding="AAD_punct", index=18}
                                // tokens.get(j).getTrimmedEncoding()="AACDD", "AAD"
                                // pattern_tree = ["AACDD", "AAD"]

                                // "hall", patternLength=2
                                // i = 1759
                                // tokens.get(j) = {word:"hall", lemma="hall", encoding="AD_nmod", index=8},
                                //                 {word:".", lemma=".", encoding="AE_punct", index=56}
                                // tokens.get(j).getTrimmedEncoding()="AD", "AE"
                                // pattern_tree = ["AD", "AE"]
                                // i = 1767
                                // tokens.get(j) = {word:"hall", lemma="hall", encoding="AD_nmod", index=8},
                                //                 {word:"for", lemma="for", encoding="ADDA_case", index=9}
                                // tokens.get(j).getTrimmedEncoding()="AD", "ADDA"
                                // pattern_tree = ["AD", "ADDA"]
                            }

                            String pattern_root = min_characters(pattern_tree);
                            // i=1390 pattern_tree=["ACEA", "ACE"], then pattern_root = "ACE"
                            // i=2966 pattern_tree=["AACDD", "AAD"], then pattern_root = "AAD"
                            // i=1759 pattern_tree=["AD", "AE"], then pattern_root = "AD"
                            // i = 1767 pattern_tree=["AD", "ADDA"], then pattern_root = "AD"
                            for (String t : pattern_tree) {
                                // criteria 2 - pattern completeness in grammar: for a candidate pattern, its tokens should form a connected graph on the dependency parse tree,
                                //  i.e., ∀i, j 􏰂 n, there is a path between t_mpi and t_mpj on Tree(s)
                                // t.substring(0, t.length()-1) removes the last character of pattern entries
                                if (!pattern_tree.contains(t.substring(0,t.length()-1)) && !t.equals(pattern_root)){
                                    new_valid = 0; // if any one in pattern tree is neither root nor the first (n-1) doesn't exist in pattern_tree
                                    break;
                                }
                            }

                            updateMapCount(newdict_token, newPattern, i + 1);
                            // newdict_token {"year apart":1391, "year ,": 2967, "hall .": 1760, "hall for": 1768, }
                            updateMapCount(newvalid_pattern, newPattern, new_valid);
                            // newvalid_pattern {"year apart": 1, "year ,": 0, "hall .":0, "hall for":0, }
                        } // end if (i + 1 < tokenNumber)
                    } // end for (Integer i : positions)
                } // end if (frequency >= minimumSupport)
            } // end for (String pattern: dict_token.keySet())

            // update dict_token
            dict_token = new HashMap<>(newdict_token);
            // {"ORGANIZATION who":[2716], "the ORGANIZATION": [488, 1080, 2451, 2601, 2668], "the owner": [1395], "for wife": [1689], ...}

            // update valid_pattern
            valid_pattern = new HashMap<>(newvalid_pattern);
            //

        } // outer while loop

        logger.log(Level.INFO, "COMPLETED: Frequent Pattern Mining - Iteration " + iteration);
    }

    /*********************************************** Hierarchical PushUp ***********************************************/

    /**
     *
     * @param sentence -> List of SubSentWords objects of pushed-up or original subsentence
     * @param currentNodeIndex -> index of the current word being looked at
     * @param prevNodeIndex -> index of previous word that was being looked at
     * @param entityIndex -> Index of the entity word
     * @param potentialPatternWords -> words that will together be matched for patterns
     * @return if at least one pattern is matched or not
     */
    private static boolean treeSearch(List<SubSentWord> sentence, int currentNodeIndex, int prevNodeIndex,
                                      int entityIndex, Set<SubSentWord> potentialPatternWords){
        Set<SubSentWord> copyOfPatternWords = new TreeSet<>(potentialPatternWords);
        if (currentNodeIndex != entityIndex) {
            copyOfPatternWords.add(sentence.get(currentNodeIndex));
            if (allPattern.containsKey(returnMetaPattern(copyOfPatternWords))) return true;
        }

        boolean validPushUpScenario;
        for (int i = 65; i < 91; i++) {
            String ch = sentence.get(currentNodeIndex).getTrimmedEncoding() + (char)i;
            OptionalInt child = returnEncodeIndex(sentence, ch);
            if (child.isPresent() && (prevNodeIndex == entityIndex || child.getAsInt() != prevNodeIndex)) {
                validPushUpScenario = treeSearch(sentence, child.getAsInt(), currentNodeIndex, entityIndex, copyOfPatternWords);
                if (validPushUpScenario) {
                    return true;
                }
            }
        }
        String currentWordEncoding = sentence.get(currentNodeIndex).getTrimmedEncoding();
        if (currentWordEncoding.length() < sentence.get(prevNodeIndex).getTrimmedEncoding().length()) {
            OptionalInt parent = returnEncodeIndex(sentence, currentWordEncoding.substring(0, currentWordEncoding.length() - 1));
            if (parent.isPresent()) {
                return treeSearch(sentence, parent.getAsInt(), currentNodeIndex, entityIndex, copyOfPatternWords);
            }
        }
        return false;
    }

    // find the index of entity (using ner Types)
    private static SubSentWord identify_entityLeaves(SubSentWord rootExp, SentenceProcessor wholeSentence){
        List<SubSentWord> sentence = wholeSentence.getPushedUpSentences().getOrDefault(rootExp, wholeSentence.
                getSentenceBreakdown().get(rootExp));

        List<String> typeSet = new ArrayList<>();
        // Index of the entity in the sub-sentence
        int entityIndex = -1;
        for (int i = 0; i < sentence.size(); i++){
            SubSentWord word = sentence.get(i);
            for (String ner : nerTypes){
                if (word.getLemma().contains(ner)){
                    typeSet.add(ner);
                    entityIndex = i;
                    // If number of entities in subsentence are greater than 1, skip the identification
                    if (typeSet.size() > 1) {
                        return null;
                    }
                }
            }
        }

        if (typeSet.isEmpty()) {
            return null;
        }

        Set<SubSentWord> potentialPatternWords = new TreeSet<>();
        boolean match_set = false;

        potentialPatternWords.add(sentence.get(entityIndex));
        String currentWordEncoding = sentence.get(entityIndex).getTrimmedEncoding();
        // find parent by taking the first (n-1) of encoding
        OptionalInt parent = returnEncodeIndex(sentence, currentWordEncoding.substring(0, currentWordEncoding.length()-1));
        if (parent.isPresent()) {
            match_set = treeSearch(sentence, parent.getAsInt(), entityIndex, entityIndex, potentialPatternWords);
        }
        return match_set ? sentence.get(entityIndex) : null;
    }

    // parallel
    private static void hierarchicalPushUp(SentenceProcessor sentence) { // sentence: {sentence, sentID, sentenceBreakdown, reverseEncoding, replaceSurfaceName, pushedUpSentences}
        sentence.resetPushUp();
        // sort_leafToTop: leaf is those with longer getTrimmedEncoding, for example(ABECB is leaf and A is root)
        List<SubSentWord> sortedSubKeys = new ArrayList<>(sort_leafToTop(sentence));

        for (SubSentWord sub : sortedSubKeys) { // sub={SubSentWord}={word:"ORGANIZATION212296", lemma:"ORGANIZATION", encoding:"ABECB_nmod", index=21}
            boolean termReplaced = false;
            List<SubSentWord> subWords = new ArrayList<>(sentence.getSentenceBreakdown().get(sub)); // subWords: [{word: "ORGANIZATION212296", lemma="ORGANIZATION", encoding="ABECB_nmod", index=21}]
            if (!sentence.getReplaceSurfaceName().isEmpty()) {
                for (int i = 0; i < subWords.size(); i++) {
                    String encode = subWords.get(i).getTrimmedEncoding();
                    if (sentence.getReplaceSurfaceName().containsKey(encode)) {
                        termReplaced = true;
                        SubSentWord replacement = new SubSentWord(sentence.getReplaceSurfaceName().get(encode));
                        replacement.setEncoding(subWords.get(i).getEncoding());
                        subWords.set(i, replacement);
                    }
                }
            }
            if (termReplaced) {
                sentence.pushUpSentences(sub, subWords);
            }
            SubSentWord entityLeaf = identify_entityLeaves(sub, sentence);
            if (entityLeaf != null) {
                sentence.updateReplacedSurfaceName(sub.getTrimmedEncoding(), entityLeaf);
            }
        }
    }

    /*********************************************** Save Out Patterns ***********************************************/

    private static List<PatternCandidate> loadPatternData(int epoch) throws IOException {
        logger.log(Level.INFO, "STARTING: Meta-Patterns Loading");
        String name = (epoch == -1) ? "allPatterns" : "c_allPatterns";
        String directory = outputFolder;
        String commonSuffix = "." + noOfLines + "_" + minimumSupport;
        String suffix = (epoch == -1) ? commonSuffix + ".txt" : commonSuffix + "_" + epoch + ".txt";
        File file = new File(directory + name + suffix);
        BufferedReader patterns = new BufferedReader(new FileReader(file));
        String line = patterns.readLine();
        List<PatternCandidate> patternList = new ArrayList<>();
        while (line != null) {
            String[] data = line.split("->");
            patternList.add(new PatternCandidate(data[0], nerTypes, stopWords, Integer.parseInt(data[1])));
            line = patterns.readLine();
        }
        patterns.close();
        logger.log(Level.INFO, "COMPLETED: Meta-Patterns Loading");
        if (epoch != -1) {
            // delete temp allPattern file once it has been successfully loaded
            if (file.delete()) logger.log(Level.INFO, "DELETED: Epoch file " + file.getName());
        }
        return patternList;
    }

    private static void pattern_classification(List<PatternCandidate> frequentPatterns) {
        logger.log(Level.INFO, "STARTING: Meta Pattern Classification");
        for (PatternCandidate patternCandidate : frequentPatterns.stream().sorted(
                Comparator.comparingInt(PatternCandidate::getFrequency).reversed()).collect(Collectors.toList())) {
            classifyAndAddPattern(patternCandidate);
        }
        logger.log(Level.INFO, "COMPLETED: Meta Pattern Classification");
    }

    private static List<PatternInstance> patternMatchingHelper(SentenceProcessor sentence, Map<SubSentWord, List<SubSentWord>> dict,
                                                               SubSentWord subRoot, List<List<MetaPattern>> patternList) {
        List<PatternInstance> out = new ArrayList<>();
        int multiCount = 0;
        List<SubSentWord> subSent = dict.get(subRoot);
        int entityCount = noOfEntities(subSent, nerTypes);
        for (int i = 0; i < patternList.size(); i++) {
            if (entityCount == 0 || (i == 0 && entityCount < 2)) continue;
            int endIndex = -1, startIndex = subSent.size();
            for (MetaPattern metaPattern : patternList.get(i)) {
                if (i != 0 && (multiCount > 0 || metaPattern.toString().replaceAll("[_\\-]+", " ").split(" ").length < 3)) break;
                PatternMatchIndices matchFound = check_subsequence(subSent, true, metaPattern, nerTypes);
                if (matchFound != null) {
                    int newStart = matchFound.getEntityIndices().get(0), newEnd = matchFound.getEntityIndices().get(matchFound.getEntityIndices().size() - 1);
                    boolean check1 = (i != 0) ? (newStart > endIndex) : (newStart >= endIndex);
                    boolean check2 = (i != 0) ? (newEnd < startIndex) : (newEnd <= startIndex);
                    if (check1 || check2) {
                        if (i == 0) multiCount++;
                        startIndex = Math.min(startIndex, newStart);
                        endIndex = Math.max(endIndex, newEnd);
                        PatternInstance instance = new PatternInstance(sentence, subRoot, metaPattern,
                                matchFound, nerTypes);
                        out.add(instance);
                        out.addAll(instance.generateAlternatePattern());
                    }
                }
            }
        }
        return out.isEmpty() ? null : out;
    }

    private static String patternFinding(SentenceProcessor sentence, List<List<MetaPattern>> patternList) throws Exception {
        try {
            List<PatternInstance> ans = new ArrayList<>();
            List<PatternInstance> output;
            Map<SubSentWord, List<SubSentWord>> map;
            for (SubSentWord subRoot : sentence.getSentenceBreakdown().keySet()) {
                int iter = 1;
                if (sentence.getPushedUpSentences().containsKey(subRoot)) {
                    iter++;
                }
                while (iter > 0) {
                    map = (iter == 2) ? sentence.getPushedUpSentences() : sentence.getSentenceBreakdown();
                    output = patternMatchingHelper(sentence, map, subRoot, patternList);
                    if (output != null) ans.addAll(output);
                    iter--;
                }
            }
            String answer = null;
            if (!ans.isEmpty()) {
                ans = ans.stream().sorted(Comparator.comparing(e -> e.getAllElementIndices().size())).distinct().collect(Collectors.toList());
                List<PatternInstance> finalAns = new ArrayList<>();
                boolean redundant;
                for (int i = 0; i < ans.size() - 1; i++) {
                    redundant = false;
                    for (int j = i + 1; j < ans.size(); j++) {
                        if (ans.get(j).getAllElementIndices().containsAll(ans.get(i).getAllElementIndices())) {
                            redundant = true;
                            break;
                        }
                    }
                    if (!redundant) finalAns.add(ans.get(i));
                }
                finalAns.add(ans.get(ans.size() - 1));
                answer = includeContext ? createExtractionString(finalAns) : finalAns.stream().map(PatternInstance::toString).collect(Collectors.joining(""));
            }
            return answer;
        } catch (Exception e) {
            logger.log(Level.WARNING,"ERROR in " + sentence.getSentID());
            throw new Exception(e);
        }
    }

    // parallel
    private static void savePatternMatchingResults(List<List<MetaPattern>> patternList, int start, int end, int epochNo) throws Exception{
        logger.log(Level.INFO, "STARTING: Pattern Matching");
        String outputDirectory = outputFolder;
        String suffix = "." + noOfLines + "_" + minimumSupport + "_" + epochNo + ".txt";
        BufferedWriter patternOutput = new BufferedWriter(new FileWriter(outputDirectory + "c_patternOutput" + suffix));
        for (int batchNo = start; batchNo < end; batchNo++) {
            List<SentenceProcessor> sentenceCollectorBatch = loadSentenceBreakdown(batchNo, noOfPushUps + 1);
            String[] outputArray = new String[sentenceCollectorBatch.size()];
            forkJoinPool.submit(() -> IntStream.range(0, sentenceCollectorBatch.size()).parallel()
                    .forEach(s -> {
                        try {
                            outputArray[s] = patternFinding(sentenceCollectorBatch.get(s), patternList);
                        } catch (Exception e) {
                            StringWriter errors = new StringWriter();
                            e.printStackTrace(new PrintWriter(errors));
                            logger.log(Level.WARNING, errors.toString());
                        }
                    })
            ).get();
            Arrays.stream(outputArray).forEachOrdered(s -> {
                try {
                    if (s != null) patternOutput.write(s);
                } catch (IOException e) {
                    StringWriter errors = new StringWriter();
                    e.printStackTrace(new PrintWriter(errors));
                    logger.log(Level.SEVERE, errors.toString());
                }
            });
            sentenceCollectorBatch.clear();
            logger.log(Level.INFO, "PROCESSED: Batch Iteration Number - " + batchNo);
        }
        patternOutput.close();
        logger.log(Level.INFO, "COMPLETED: Pattern Matching");
    }

    // write singlePatterns, multiPatterns and allPatterns to corresponding files
    private static void savePatternClassificationData(int epoch, SavePatternType typeOfSave) throws IOException{
        logger.log(Level.INFO, "STARTING: Saving Meta-Pattern Classification Data");
        String commonSuffix = "." + noOfLines + "_" + minimumSupport;
        String suffix = (epoch == -1) ? commonSuffix + ".txt" : commonSuffix + "_" + epoch + ".txt";
        if (typeOfSave == SavePatternType.ALL_THREE) {
            FileWriter singlePatterns = new FileWriter(outputFolder + (epoch == -1 ? "" : "c_") + "singlePatterns" + suffix);
            FileWriter multiPatterns = new FileWriter(outputFolder + (epoch == -1 ? "" : "c_") + "multiPatterns" + suffix);
            writePatternsToFile(new BufferedWriter(singlePatterns), singlePattern);
            writePatternsToFile(new BufferedWriter(multiPatterns), multiPattern);
        }
        FileWriter allPatterns = new FileWriter(outputFolder + (epoch == -1 ? "" : "c_") + "allPatterns" + suffix);
        writePatternsToFile(new BufferedWriter(allPatterns), allPattern);
        logger.log(Level.INFO, "COMPLETED: Saving Meta-Pattern Classification Data");
    }

    /*****************************************************************************************************/
    /***********************************************  CALL ***********************************************/
    /*****************************************************************************************************/

    public static void call() {
        try {
            initialization();
            if (!load_sentenceBreakdownData) {
//                buildDictionary();
                buildSentences();
            }
            int endBatchExclusive = countSavedBatches(inputFolder + "ProcessedInput/" + batchSize + "/", noOfLines, startBatch);
            if (endBatchExclusive == 0) throw new IOException("Sentence Breakdown Data does not exist.");

            int noOfEpochs = (endBatchExclusive - startBatch - 1)/splitSize + 1;
            // noOfEpochs = 1
            for (int epoch = 0; epoch < noOfEpochs; epoch++) {
                int epochStart = startBatch + (epoch * splitSize); // epochStart = 0
                int epochEndExclusive = Math.min(startBatch + ((epoch + 1) * splitSize), endBatchExclusive); // epochEndExclusive = 1

                if (noOfEpochs > 1) logger.log(Level.INFO, "STARTING: Epoch No." + (epoch + 1) + "\n");
                int prevPatternCount = 0;
                int iterations = 1;

                // the first time we do frequent pattern mining, there is no elements in sentence.getPushedUpSentences().keySet()
                List<List<MetaPattern>> patternList = new ArrayList<>();
                // load_metapatternData=false
                if (load_metapatternData) {
                    pattern_classification(loadPatternData(-1));
                    noOfPushUps = 1;
                } else {
                    logger.log(Level.INFO, "STARTING: Pattern Distillation");
                    frequent_pattern_mining(iterations, epochStart, epochEndExclusive);
                }

                // allPattern=28 {"history ORGANIZATION": 2, "ORGANIZATION chief executive": 4, "PERSON be bear": 6, "chief executive PERSON": 2, "PERSON to LOCATION": 3, ...}

                // noOfPushUps = 3, prePatternCount=0, allPattern.size()=28
                while (iterations <= noOfPushUps && prevPatternCount < allPattern.size()) {
                    prevPatternCount = allPattern.size(); //prePatternCount=28
                    logger.log(Level.INFO, "STARTING: Hierarchical PushUps - Iteration " + iterations);
                    // batchNo=0, epochStart=0, epochEndExclusive=1
                    for (int batchNo = epochStart; batchNo < epochEndExclusive; batchNo++) {
                        List<SentenceProcessor> sentenceCollectorBatch = loadSentenceBreakdown(batchNo, iterations == 1 ? -1 : iterations);
                        // sentenceCollectorBatch (size=60)
                        // sentenceCollectorBatch[0]: sentence, sentID, sentenceBreakdown, reverseEncoding, replaceSurfaceName, pushedUpSentences
                        forkJoinPool.submit(() -> sentenceCollectorBatch.parallelStream().forEach(Sempad::hierarchicalPushUp)).get();
                        saveSentenceBreakdown(batchNo, sentenceCollectorBatch, iterations + 1);
                        sentenceCollectorBatch.clear();
                        logger.log(Level.INFO, "PROCESSED: Batch Iteration Number - " + batchNo);
                    }

                    logger.log(Level.INFO, "COMPLETED: Hierarchical PushUps - Iteration " + iterations++);
                    if (!load_metapatternData) frequent_pattern_mining(iterations, epochStart, epochEndExclusive);
                } // end while (iterations <= noOfPushUps && prevPatternCount < allPattern.size())

                if (iterations <= noOfPushUps) noOfPushUps = iterations - 1;
                if (noOfPushUps == 0) {
                    throw new IOException("No patterns found.");
                }

                savePatternClassificationData(epoch, SavePatternType.ALL);
                patternList.add(returnSortedPatternList(multiPattern));
                if (includeContext) patternList.add(returnSortedPatternList(singlePattern));
                savePatternMatchingResults(patternList, epochStart, epochEndExclusive, epoch);
            } //end for (int epoch = 0; epoch < noOfEpochs; epoch++)

            logger.log(Level.INFO, "CREATING: Combined output file from all epochs.");
            String suffix = "c_patternOutput." + noOfLines + "_" + minimumSupport + ".txt";
            BufferedWriter bw = new BufferedWriter(new FileWriter(outputFolder + suffix));

            // Aggregating pattern classification data and pattern output data from each epoch (actuallly we have only epoch here)
            for (int epoch = 0; epoch < noOfEpochs; epoch++) {
                suffix = "c_patternOutput." + noOfLines + "_" + minimumSupport + "_" + epoch + ".txt";
                File patternOutputFile = new File(outputFolder + suffix);
                BufferedReader br = new BufferedReader(new FileReader(patternOutputFile));
                String line = br.readLine();

                while (line != null) {
                    bw.write(line + "\n");
                    line = br.readLine();
                }

                clearPatterns();
                pattern_classification(loadPatternData(epoch));
                br.close();
                if (patternOutputFile.delete()) logger.log(Level.INFO, "DELETED: Epoch file " + patternOutputFile.getName());
            }

            savePatternClassificationData(-1, SavePatternType.ALL_THREE);
            bw.close();
            logger.log(Level.INFO, "CREATED: Combined output file from all epochs.");
        } catch (Exception e) {
            StringWriter errors = new StringWriter();
            e.printStackTrace(new PrintWriter(errors));
            logger.log(Level.SEVERE, errors.toString());
        } finally {
            if (forkJoinPool != null) {
                forkJoinPool.shutdown();
            }
        }
    }
}