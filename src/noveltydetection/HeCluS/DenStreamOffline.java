package noveltydetection.HeCluS;

import java.io.*;
import java.util.ArrayList;

import moa.cluster.Clustering;
import weka.core.DenseInstance;
import weka.core.Instance;

public class DenStreamOffline {

    private ArrayList<Integer> clusteringResul[];
    private int clusterExamples[];

//*******************************************************************************   
//******************** Clustering ***********************************************
//*******************************************************************************    
    public Clustering DenStream(String lblClass, String filename, double epsilon, double mu, double beta, int minPoints) throws IOException {
        BufferedReader fileIn;
        Clustering microClusters;
        String line;

        WithDBSCAN algClustream = new WithDBSCAN();
        algClustream.prepareForUse();
        algClustream.epsilon = epsilon;
        algClustream.mu = mu;
        algClustream.beta = beta;
        algClustream.minPoints = minPoints;

        //read the data set and executing clustream algorithm
        int lineSize;
        fileIn = new BufferedReader(new FileReader(new File(filename + lblClass)));
        double data[];
        int cont = 0;
        while ((line = fileIn.readLine()) != null) {
            String linhaarqString[] = line.split(",");
            lineSize = linhaarqString.length - 1;
            data = new double[lineSize];
            cont++;
            for (int i = 0; i < lineSize; i++) {
                data[i] = Double.parseDouble(linhaarqString[i]);
            }
            Instance inst = new DenseInstance(1, data);
            algClustream.trainOnInstanceImpl(inst);
        }
        fileIn.close();

        //obtain the micro-clusters  
        microClusters = algClustream.getMicroClusteringResult();
        ArrayList<ArrayList<Integer>> grupos;

        if (microClusters.size() >= 1) {
            grupos = ((WithDBSCAN) algClustream).getClusterExamples();

            clusterExamples = new int[cont];
            int value;
            for (int i = 0; i < grupos.size(); i++) {
                //remove micro-cluster with less than 3 examples
                if (((MicroClusterMOA) microClusters.get(i)).getN() < 1) {
                    value = -1;
                } else {
                    value = i;
                }

                for (int j = 0; j < grupos.get(i).size(); j++) {
                    clusterExamples[grupos.get(i).get(j)] = value; //i+1;
                }
                if (((MicroClusterMOA) microClusters.get(i)).getN() < 1) {
                    microClusters.remove(i);
                    grupos.remove(i);
                    i--;
                }
            }
        }

        return microClusters;
    }

//*******************************************************************************   
//******************** getClusterExamples ***************************************
//******************************************************************************* 	
    public int[] getClusterExamples() {
        return clusterExamples;
    }

//*******************************************************************************   
//******************** Clustering ***********************************************
//******************************************************************************* 	
    public int[] getClusteringResults() {
        int resultado[];
        int nroelem = 0;
        for (ArrayList<Integer> clusteringResul1 : clusteringResul) {
            nroelem += clusteringResul1.size();
        }
        resultado = new int[nroelem];

        for (int i = 0; i < clusteringResul.length; i++) {
            for (int j = 0; j < clusteringResul[i].size(); j++) {
                resultado[clusteringResul[i].get(j)] = i + 1;
            }
        }
        return resultado;
    }
}
