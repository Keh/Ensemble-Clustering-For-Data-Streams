package noveltydetection.HeCluS;

import java.io.*;
import java.util.ArrayList;

import moa.cluster.Clustering;
import weka.core.DenseInstance;
import weka.core.Instance;

public class ClustreeOffline {

    //*******************************************************************************   
    //******************** Clustering ***********************************************
    //*******************************************************************************    
    public Clustering TreeStream(String lblClass, String filename, int arvore) throws IOException {
        
        BufferedReader fileIn;
        Clustering microClusters = null;
        String line;

        ClusTree algClustream = new ClusTree();
        algClustream.prepareForUse();
        algClustream.height=arvore;

        //read the data set and executing clustream algorithm
        int lineSize = 0;
        fileIn = new BufferedReader(new FileReader(new File(filename + lblClass)));
        double data[] = null;
        while ((line = fileIn.readLine()) != null) {
            String linhaarqString[] = line.split(",");
            lineSize = linhaarqString.length - 1;
            data = new double[lineSize];
            for (int i = 0; i < lineSize; i++) {
                data[i] = Double.parseDouble(linhaarqString[i]);
            }
            Instance inst = new DenseInstance(1, data);
            algClustream.trainOnInstanceImpl(inst);
        }
        fileIn.close();

        //obtain the micro-clusters  
        microClusters = algClustream.getMicroClusteringResult();
        
        return microClusters;
    }

    
}
