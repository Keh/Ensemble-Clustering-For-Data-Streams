//essa versao difere da anterior pq agora coloquei o calculo das medidas de avaliacao online
//antes eu processava tudo primeiro e depois fazia as avaliacoes

package noveltydetection.HeCluS;

import java.util.*;

import java.io.*;
import java.math.BigDecimal;
import moa.cluster.CFCluster;
import moa.cluster.Clustering;

import weka.core.*;

public class MINAS {

	private String filenameOffline; // filename used in offline phase
	private String filenameOnline; // filename used in online phase
	private String outputDirectory; // name of the output directory
	private FileWriter fileOut; // name of the output file --> classification results
	private FileWriter addInfo;
	private FileWriter fileOutClasses; // name of the output file --> problem classes

	private double threshold; // threshold: used to separate novelty patterns from extensions
	private int numMicroClusters = 100; // number of micro-clusters used by the clustering algorithm
	private int numMinExCluster; // minimum number of examples in a valid cluster
	private int numExNoveltyDetection = 2000; // minimum number of examples in the unknown memory to execute the ND
												// procedure

	// info about each clustering technique
	private String alg_C = "clustream";
	private String alg_D = "denstream";
	private String alg_T = "clustree";

	private int evaluationType; // evaluation type: 0- proposed methodology 1- methodology used in the
								// literature
	private String currentState; // current state of the system: phase offline or online
	private int lastCheck = 0; // last time the clustering algorithm was executed
	private int timestamp = -1; // current timestamp
	private boolean flagMicro; // indicate if will be used the option clustream + kmeans

	private ArrayList<MicroClusterC> sleepC; // memory sleep: used to store old micro-clusters
	private ArrayList<MicroClusterD> sleepD; // memory sleep: used to store old micro-clusters
	private ArrayList<MicroClusterT> sleepT; // memory sleep: used to store old micro-clusters

	// list of sleep
	private ArrayList<ArrayList<MicroClusterC>> SleepModeMemC;
	private ArrayList<ArrayList<MicroClusterD>> SleepModeMemD;
	private ArrayList<ArrayList<MicroClusterT>> SleepModeMemT;

	private ArrayList<String[]> dataUnk; // memory unknown: used to store the examples classified as unknown
	private ArrayList<Integer> timestamp_data_unk; // timestamp of the unknwon examples
	private int windowSize = 4000; // window size: used to eliminate old micro-clusters and old unknown examples

	// model
	private ArrayList<ArrayList<MicroClusterC>> modelC; // decision model used to classify new examples
	private ArrayList<ArrayList<MicroClusterD>> modelD; // decision model used to classify new examples
	private ArrayList<ArrayList<MicroClusterT>> modelT; // decision model used to classify new examples

	ArrayList<Integer> numPart;

	static int numC = 0;
	static int numD = 0;
	static int numT = 0;

	// Para o DENSTREAM
	static double epsilon = 0.02;
	static double mu = 1;
	static double beta = 0.2;
	static int minPoints = 1;

	// Para o CLUSTREE
	static int alturArvore = 7;

//******************************************************************************   
//******************** Methods Set *********************************************
//****************************************************************************** 
	public void numMinExCluster(int par_minexcl) {
		this.numMinExCluster = par_minexcl;
	}

//*******************************************************************************   
//******************** Construtor ***********************************************
//********************************************************************************    
	public MINAS(String fileOff, String fileOnl, String outDirec, int nMicro,
			double threshold, int numExNoveltyDetection, int windowSize, 
			int numC, int numD, int numT, double epsilon, double mu, double beta, int minPoints, int arv) 
					throws IOException {
		this.filenameOffline = fileOff;
		this.filenameOnline = fileOnl;
		this.outputDirectory = outDirec + alg_C+"_" + alg_D+"_" + alg_T +"_numMicro" + nMicro + "_numExDN"
				+ numExNoveltyDetection + "_t" + threshold + "_f" + windowSize;
		
		this.alg_C = "clustream";
		this.alg_D = "denstream";
		this.alg_T = "clustree";
		this.evaluationType = 0;
		this.flagMicro = true;
		
		this.threshold = threshold;		
		this.numMicroClusters = nMicro;
		this.numExNoveltyDetection = numExNoveltyDetection;
		this.windowSize = windowSize;
		
		createDirectory();

		this.numC = numC;
		this.numD = numD;
		this.numT = numT;
		
		this.epsilon = epsilon;
		this.mu = mu;
		this.beta = beta;
		this.minPoints = minPoints;
		this.alturArvore = arv;
		
	}

//*******************************************************************************   
//********************createDirectory********************************************
//******************************************************************************** 
	// create an output directory
	public void createDirectory() {
		File dir = new File(outputDirectory);
		if (dir.exists()) {
			dir.delete();
		} else if (dir.mkdirs()) {
			System.out.println("Directory created successfully");
		}
	}

//*******************************************************************************   
//********************createFileOut**********************************************
//******************************************************************************* 
	// create output files:
	// fileOut: classification result for each example
	// fileOutClasses: problem classes
	public void createFileOut() throws IOException {
		String fileNameOut = outputDirectory + "\\results";
		String fileNameOutClasses = outputDirectory + "\\results" + "Classes";
		String fileNameOut2 = outputDirectory + "\\Info";
		fileOut = new FileWriter(new File(fileNameOut), false);
		fileOutClasses = new FileWriter(new File(fileNameOutClasses), false);
		addInfo = new FileWriter(new File(fileNameOut2), false);
		fileOut.write("Results");
		fileOut.write("\n\n \n\n");
	}
//*******************************************************************************   
//******************** offline **************************************************
//*******************************************************************************     

	public ArrayList<String> offline(ArrayList<Integer> numPart) throws IOException {
		// create one training file for each problem class
		SeparateFile sp = new SeparateFile();
		ArrayList<String> trainingClasses = sp.separateLabeledFile(filenameOffline);
		System.out.print("\n Classes for the training phase (offline): ");
		System.out.println("" + trainingClasses.size());

		modelC = new ArrayList<ArrayList<MicroClusterC>>();
		modelD = new ArrayList<ArrayList<MicroClusterD>>();
		modelT = new ArrayList<ArrayList<MicroClusterT>>();

		// model C
		for (int i = 0; i < numPart.get(0); i++) {
			// generate a set of micro-clusters for each class from the training set
			ArrayList<MicroClusterC> clusterSetC = null;
			ArrayList<MicroClusterC> model = new ArrayList<MicroClusterC>();

			for (String cl : trainingClasses) {
				System.out.println("Classe " + cl);
				int[] clusteringResult = new int[1];
				// if the clustering algorithm is clustream
				if (alg_C.compareTo("clustream") == 0) {
					clusterSetC = criamodeloCluStream(filenameOffline, numMicroClusters, cl, clusteringResult);
				}
				for (int j = 0; j < clusterSetC.size(); j++) {
					model.add(clusterSetC.get(j));
				}
			}
			modelC.add(model);
		}

		// model D
		for (int i = 0; i < numPart.get(1); i++) {
			// if the clustering algorithm is denstream
			ArrayList<MicroClusterD> clusterSetD = null;
			ArrayList<MicroClusterD> model = new ArrayList<MicroClusterD>();
			for (String cl : trainingClasses) {
				System.out.println("Classe " + cl);
				int[] clusteringResult = new int[1];
				if (alg_D.compareTo("denstream") == 0) {
					clusterSetD = createModelDenStream(filenameOffline, cl, clusteringResult, epsilon, mu, beta,
							minPoints);
					System.out.println("clusterSetD " + clusterSetD.size());
				}
				for (int j = 0; j < clusterSetD.size(); j++) {
					model.add(clusterSetD.get(j));
				}
			}
			modelD.add(model);
		}

		// model T
		for (int i = 0; i < numPart.get(2); i++) {
			// if the clustering algorithm is clustree
			ArrayList<MicroClusterT> clusterSetT = null;
			ArrayList<MicroClusterT> model = new ArrayList<MicroClusterT>();

			for (String cl : trainingClasses) {
				System.out.println("Classe " + cl);
				int[] clusteringResult = new int[1];
				if (alg_T.compareTo("clustree") == 0) {
					clusterSetT = createModelTree(filenameOffline, alturArvore, cl, clusteringResult);
				}
				for (int j = 0; j < clusterSetT.size(); j++) {
					model.add(clusterSetT.get(j));
				}
			}
			modelT.add(model);

		}
		System.out.println("size model - clustream: " + modelC.size() + "\tdenstream: " + modelD.size() + "\tclustree: "
				+ modelT.size());

		for (int i = 0; i < modelC.size(); i++)
			System.out.println("size model 1 - clustream: " + modelC.get(i).size() + " i " + i);
		for (int i = 0; i < modelD.size(); i++)
			System.out.println("size model 2 - denstream: " + modelD.get(i).size() + " i " + i);
		for (int i = 0; i < modelT.size(); i++)
			System.out.println("size model 3 - clustree: " + modelT.get(i).size() + " i " + i);

		// store in the fileOutClasses the class label used in the training phase
		// offline
		for (int g = 0; g < trainingClasses.size(); g++) {
			fileOutClasses.write(trainingClasses.get(g));
			fileOutClasses.write("\r\n");
		}
		fileOutClasses.close();
		return trainingClasses;
	}

//*******************************************************************************   
//******************** round ****************************************************
//********************************************************************************      
	public static double round(double d, int decimalPlace) {
		// see the Javadoc about why we use a String in the constructor
		// http://java.sun.com/j2se/1.5.0/docs/api/java/math/BigDecimal.html#BigDecimal(double)
		BigDecimal bd = new BigDecimal(Double.toString(d));
		bd = bd.setScale(decimalPlace, BigDecimal.ROUND_HALF_UP);
		return bd.doubleValue();
	}

//*******************************************************************************   
//******************** buildTextFileOut *****************************************
//*******************************************************************************
	// build a message to be stored in the OutputFile
	public String buildTextFileOut(String retorno[]) {
		String text = "";

		if (retorno[1].equals("normal")) {
			text = "C " + retorno[0];
		} // example belongs to an extension of a known class
		else if (retorno[1].equals("ext")) {
			text = "ExtCon " + retorno[0];
		}
		// example belongs to a novelty pattern
		if (retorno[1].equals("nov")) {
			text = "N " + retorno[0];
		}
		// exemple belongs to an extension of a novelty pattern
		if (retorno[1].equals("extNov")) {
			text = "ExtNov " + retorno[0];
		}
		return text;
	}

//*******************************************************************************   
//******************** classify *************************************************
//********************************************************************************
	// classify each new example
	public void classify(Evaluator av, String[] data, String information, ArrayList<String> noveltyPatterns,
			ArrayList<Integer> numPart) throws IOException {
		int sizeExample = data.length;
		int count_k;
		double[] data_double = new double[sizeExample - 1];
		for (int j = 0; j < sizeExample - 1; j++) {
			data_double[j] = Double.valueOf(data[j]);
		}

		// identify if a new example belongs to a known class (if it is inside of one of
		// the micro-clusters of the decision model)
		String[] result = identifyExample(data_double);

		if (result != null) {
			// ********** Example explained by the current decision model
			String textOut = null;
			String text = buildTextFileOut(result);
			textOut = information + text;

			// escrevendo no arquivo de saida
			fileOut.write(textOut);
			fileOut.write("\n");
			av.addElementConfusionMatrix(textOut);
		} else {
			// ********** Example not explained by the current decision model: marked as
			// unknown
			information = information + "não sei ";
			fileOut.write(information);
			fileOut.write("\n");
			av.addElementConfusionMatrix(information);

			// adding the example to the unkown memory
			dataUnk.add(data);
			timestamp_data_unk.add(timestamp);

			// ********** Searching for new valid micro-clusters created from unknown
			// examples
			count_k = numMicroClusters;
			Random rand = new Random();

			if ((dataUnk.size() >= numExNoveltyDetection) && (lastCheck + numExNoveltyDetection < timestamp)) {
				numMinExCluster = (int) ((float) dataUnk.size() / count_k);
				// count_k = (int)((float)data_unk.size()/par_minexcl);
				lastCheck = timestamp;
				Hashtable mat = new Hashtable();
				// for each candidate micro-cluster
				if (alg_C.compareTo("clustream") == 0) {
					ArrayList<ArrayList<MicroClusterC>> model_unkC = new ArrayList<ArrayList<MicroClusterC>>();
					ArrayList<int[]> removeExamplesC = new ArrayList<int[]>();
					for (int p = 0; p < numPart.get(0); p++) {
						// normal
						removeExamplesC.add(new int[dataUnk.size()]);
						model_unkC.add(createModelFromExamples(dataUnk, alg_C, count_k, removeExamplesC.get(p)));

//						// rand bagging
//						ArrayList<String[]> dataUnk2 = new ArrayList<String[]>();
//						ArrayList<Integer> index = new ArrayList<Integer>();
//						int bag = 0;
//						for (int b = 0; b < (dataUnk.size()/2); b++) {
//							bag = rand.nextInt(dataUnk.size());					
//							index.add(bag);
//						}
//
//						Collections.sort(index);
//						
//						for(int b = 0; b < (index.size()); b++ ) {
//							dataUnk2.add(dataUnk.get(index.get(b)));				
//						}
//
//						removeExamplesC.add(new int[dataUnk2.size()]);
//						model_unkC.add(createModelFromExamples(dataUnk2, alg_C, count_k, removeExamplesC.get(p)));

					}

//				model_unkG = combiner(numPart, model_unkG);

					int max = 0;
					for (int p = 0; p < numPart.get(0); p++) {
						if (max < model_unkC.get(p).size()) {
							max = model_unkC.get(p).size();
						}
					}
//					System.out.println("max " + max);

					
//					System.out.println("^^^antes^^^^^");
					int remove = 0;

					for (int p = 0; p < model_unkC.size(); p++) {

//					System.out.println("^^^^^^^^^");

						for (int temp_count_k = 0; temp_count_k < model_unkC.get(p).size(); temp_count_k++) {
							MicroClusterC model_unk_aux = model_unkC.get(p).get(temp_count_k);
							ArrayList<MicroClusterC> newModels = new ArrayList<MicroClusterC>();

							if ((model_unk_aux != null) && (model_unk_aux.getWeight() >= numMinExCluster)) {
								if (clusterValidationSilhouette(model_unk_aux, p) == true) {

									// ********** The new micro-cluster is valid ****************
									// creating a partial confusion matrix in order to move the unknown examples to
									// the corresponding classes
									mat = new Hashtable();
									// p <- remove
									if (p == 0) {
										for (int count_remove = 0; count_remove < removeExamplesC
												.get(p).length; count_remove++) {
											// mark the examples to be removed with the label -2
											if (removeExamplesC.get(p)[count_remove] == temp_count_k) {
												removeExamplesC.get(p)[count_remove] = -2;

												if ((mat.get(
														"C " + dataUnk.get(count_remove)[sizeExample - 1])) == null) {
													mat.put("C " + dataUnk.get(count_remove)[sizeExample - 1], 1);
												} else {
													int valor = ((Integer) mat
															.get("C " + dataUnk.get(count_remove)[sizeExample - 1]));
													mat.put("C " + dataUnk.get(count_remove)[sizeExample - 1],
															valor + 1);
												}

											}
										}
									}

									// identify the closer micro-cluster to the new valid micro-cluster
									ArrayList<String> ret_func = identifyCloserMicroClusterC(model_unk_aux, threshold,
											p);
									double distance = Double.parseDouble(ret_func.get(0));
									double thresholdValue = Double.parseDouble(ret_func.get(1));
									String category = ret_func.get(2);
									String classLabel = ret_func.get(3);

									// if the distance to the closer micro-cluster is less than a threshold
									String textoArq = "";
									String word = "";
//								if()
									if (distance < thresholdValue) {
										// extension
										if ((category.equalsIgnoreCase("normal"))
												|| (category.equalsIgnoreCase("ext"))) {
											// extension of a class learned offline
											newModels.add(model_unk_aux);
											word = "ext";
											updateModelC(model_unk_aux, classLabel, "ext", p);
											textoArq = "Thinking " + "Extensao: " + "C " + classLabel + " - "
													+ (int) model_unk_aux.getWeight() + " exemplos";

										} else {
											// extension of a novelty pattern
											newModels.add(model_unk_aux);
											word = "extNov";
											updateModelC(model_unk_aux, classLabel, "extNov", p);
											textoArq = "Thinking " + "ExtensaoNovidade: " + "N " + classLabel + " - "
													+ (int) model_unkC.get(p).get(temp_count_k).getWeight()
													+ " exemplos";
										}
									} else {
										// novelty pattern
										newModels.add(model_unk_aux);
										word = "nov";
										updateModelC(model_unk_aux, Integer.toString(noveltyPatterns.size() + 1), "nov",
												p);
										// if the new novelty patter do no intercept any other micro-cluster
										if (modelC.get(p).get(modelC.get(p).size() - 1).getLabelClass()
												.compareToIgnoreCase(
														Integer.toString(noveltyPatterns.size() + 1)) == 0) {
											// if the new valid cluster is a recurrence of known concept
											if (verifyRecurrenceC(model_unk_aux, p)) {
												if (model_unk_aux.getCategory().compareToIgnoreCase("ext") == 0) {
													textoArq = "Thinking " + "Extensao:" + "C "
															+ modelC.get(p).get(modelC.get(p).size() - 1)
																	.getLabelClass()
															+ " - "
															+ (int) model_unkC.get(p).get(temp_count_k).getWeight()
															+ " exemplos";
												} else {
													textoArq = "Thinking " + "ExtensaoNovidade: " + "N "
															+ modelC.get(p).get(modelC.get(p).size() - 1)
																	.getLabelClass()
															+ " - "
															+ (int) model_unkC.get(p).get(temp_count_k).getWeight()
															+ " exemplos";
												}
											} else {
												// novelty pattern was detected
												noveltyPatterns.add(noveltyPatterns.size() + 1 + "");
												textoArq = "ThinkingNov: " + "Novidade " + (noveltyPatterns.size())
														+ " - " + (int) model_unkC.get(p).get(temp_count_k).getWeight()
														+ " exemplos";
											}
										} else {
											textoArq = "Thinking " + "ExtensaoNovidade: " + "N "
													+ modelC.get(p).get(modelC.get(p).size() - 1).getLabelClass()
													+ " - " + (int) model_unkC.get(p).get(temp_count_k).getWeight()
													+ " exemplos";
										}
									}

									fileOut.write(textoArq);
									fileOut.write("\n");
									av.addColumnConfusionMatrix(textoArq, mat);
								} // end of valid cluster

							} // end of mininum number of examples

						} // end of each valid cluster
						if (max == model_unkC.get(p).size())
							remove++;

					}

					// MODEL D

					if (alg_D.compareTo("denstream") == 0) {
						ArrayList<ArrayList<MicroClusterD>> model_unkD = new ArrayList<ArrayList<MicroClusterD>>();
						ArrayList<int[]> removeExamplesD = new ArrayList<int[]>();
						for (int p = 0; p < numPart.get(1); p++) {
							// normal
							removeExamplesD.add(new int[dataUnk.size()]);
							model_unkD.add(createModelFromExamplesD(dataUnk, count_k, removeExamplesD.get(p)));

//							// rand bagging
//							ArrayList<String[]> dataUnk2 = new ArrayList<String[]>();
//							ArrayList<Integer> index = new ArrayList<Integer>();
//							int bag = 0;
//							for (int b = 0; b < (dataUnk.size()/2); b++) {
//								bag = rand.nextInt(dataUnk.size());					
//								index.add(bag);
//							}
							//
//							Collections.sort(index);
//							
//							for(int b = 0; b < (index.size()); b++ ) {
//								dataUnk2.add(dataUnk.get(index.get(b)));				
//							}
							//
//							removeExamplesC.add(new int[dataUnk2.size()]);
//							model_unkC.add(createModelFromExamples(dataUnk2, alg_C, count_k, removeExamplesC.get(p)));

						}

//					model_unkG = combiner(numPart, model_unkG);

						max = 0;
						for (int p = 0; p < numPart.get(1); p++) {
							if (max < model_unkD.get(p).size()) {
								max = model_unkD.get(p).size();
							}
						}

						mat = new Hashtable();
						remove = 0;

						for (int p = 0; p < model_unkD.size(); p++) {

							for (int temp_count_k = 0; temp_count_k < model_unkD.get(p).size(); temp_count_k++) {
								MicroClusterD model_unk_aux = model_unkD.get(p).get(temp_count_k);
								ArrayList<MicroClusterD> newModels = new ArrayList<MicroClusterD>();

								if ((model_unk_aux != null) && (model_unk_aux.getWeight() >= 0)) {
									if (clusterValidationSilhouetteD(model_unk_aux, p) == true) {

										// ********** The new micro-cluster is valid ****************
										// creating a partial confusion matrix in order to move the unknown examples to
										// the corresponding classes
										// O QUE FAZER COM ISSO?????
										mat = new Hashtable();
										// p <- remove
//										if (p == 0) {
//											for (int count_remove = 0; count_remove < removeExamplesD
//													.get(p).length; count_remove++) {
//												// mark the examples to be removed with the label -2
//												if (removeExamplesD.get(p)[count_remove] == temp_count_k) {
//													removeExamplesD.get(p)[count_remove] = -2;
//
//													if ((mat.get("C "
//															+ dataUnk.get(count_remove)[sizeExample - 1])) == null) {
//														mat.put("C " + dataUnk.get(count_remove)[sizeExample - 1], 1);
//													} else {
//														int valor = ((Integer) mat.get(
//																"C " + dataUnk.get(count_remove)[sizeExample - 1]));
//														mat.put("C " + dataUnk.get(count_remove)[sizeExample - 1],
//																valor + 1);
//													}
//
//												}
//											}
//										}

										// identify the closer micro-cluster to the new valid micro-cluster
										ArrayList<String> ret_func = identifyCloserMicroClusterD(model_unk_aux,
												threshold, p);
										double distance = Double.parseDouble(ret_func.get(0));
										double thresholdValue = Double.parseDouble(ret_func.get(1));
										String category = ret_func.get(2);
										String classLabel = ret_func.get(3);

										// if the distance to the closer micro-cluster is less than a threshold
										String textoArq = "";
										String word = "";
//									if()
										if (distance < thresholdValue) {
											// extension
											if ((category.equalsIgnoreCase("normal"))
													|| (category.equalsIgnoreCase("ext"))) {
												// extension of a class learned offline
												newModels.add(model_unk_aux);
												word = "ext";
												updateModelD(model_unk_aux, classLabel, "ext", p);
												textoArq = "Thinking " + "Extensao: " + "C " + classLabel + " - "
														+ (int) model_unk_aux.getWeight() + " exemplos";

											} else {
												// extension of a novelty pattern
												newModels.add(model_unk_aux);
												word = "extNov";
												updateModelD(model_unk_aux, classLabel, "extNov", p);
												textoArq = "Thinking " + "ExtensaoNovidade: " + "N " + classLabel
														+ " - " + (int) model_unkD.get(p).get(temp_count_k).getWeight()
														+ " exemplos";
											}
										} else {
											// novelty pattern
											newModels.add(model_unk_aux);
											word = "nov";
											updateModelD(model_unk_aux, Integer.toString(noveltyPatterns.size() + 1),
													"nov", p);
											// if the new novelty patter do no intercept any other micro-cluster
											if (modelD.get(p).get(modelD.get(p).size() - 1).getLabelClass()
													.compareToIgnoreCase(
															Integer.toString(noveltyPatterns.size() + 1)) == 0) {
												// if the new valid cluster is a recurrence of known concept
												if (verifyRecurrenceD(model_unk_aux, p)) {
													if (model_unk_aux.getCategory().compareToIgnoreCase("ext") == 0) {
														textoArq = "Thinking " + "Extensao:" + "C "
																+ modelD.get(p).get(modelD.get(p).size() - 1)
																		.getLabelClass()
																+ " - "
																+ (int) model_unkD.get(p).get(temp_count_k).getWeight()
																+ " exemplos";
													} else {
														textoArq = "Thinking " + "ExtensaoNovidade: " + "N "
																+ modelD.get(p).get(modelD.get(p).size() - 1)
																		.getLabelClass()
																+ " - "
																+ (int) model_unkD.get(p).get(temp_count_k).getWeight()
																+ " exemplos";
													}
												} else {
													// novelty pattern was detected
													noveltyPatterns.add(noveltyPatterns.size() + 1 + "");
													textoArq = "ThinkingNov: " + "Novidade " + (noveltyPatterns.size())
															+ " - "
															+ (int) model_unkD.get(p).get(temp_count_k).getWeight()
															+ " exemplos";
												}
											} else {
												textoArq = "Thinking " + "ExtensaoNovidade: " + "N "
														+ modelD.get(p).get(modelD.get(p).size() - 1).getLabelClass()
														+ " - " + (int) model_unkD.get(p).get(temp_count_k).getWeight()
														+ " exemplos";
											}
										}

										fileOut.write(textoArq);
										fileOut.write("\n");
										av.addColumnConfusionMatrix(textoArq, mat);
									} // end of valid cluster

								} // end of mininum number of examples

							} // end of each valid cluster
							if (max == model_unkD.get(p).size())
								remove++;

						}
					}
					
					// MODEL T
					if (alg_C.compareTo("clustree") == 0) {
						ArrayList<ArrayList<MicroClusterT>> model_unkT = new ArrayList<ArrayList<MicroClusterT>>();
						ArrayList<int[]> removeExamplesT = new ArrayList<int[]>();
						for (int p = 0; p < numPart.get(2); p++) {
							// normal
							removeExamplesT.add(new int[dataUnk.size()]);
							model_unkT.add(createModelFromExamplesT(dataUnk, alturArvore, removeExamplesT.get(p)));

//							// rand bagging
//							ArrayList<String[]> dataUnk2 = new ArrayList<String[]>();
//							ArrayList<Integer> index = new ArrayList<Integer>();
//							int bag = 0;
//							for (int b = 0; b < (dataUnk.size()/2); b++) {
//								bag = rand.nextInt(dataUnk.size());					
//								index.add(bag);
//							}
							//
//							Collections.sort(index);
//							
//							for(int b = 0; b < (index.size()); b++ ) {
//								dataUnk2.add(dataUnk.get(index.get(b)));				
//							}
							//
//							removeExamplesC.add(new int[dataUnk2.size()]);
//							model_unkC.add(createModelFromExamples(dataUnk2, alg_C, count_k, removeExamplesC.get(p)));

						}

//					model_unkG = combiner(numPart, model_unkG);

						max = 0;
						for (int p = 0; p < numPart.get(2); p++) {
							if (max < model_unkT.get(p).size()) {
								max = model_unkT.get(p).size();
							}
						}

						
						for (int p = 0; p < model_unkT.size(); p++) {

//						System.out.println("^^^^^^^^^");

							for (int temp_count_k = 0; temp_count_k < model_unkT.get(p).size(); temp_count_k++) {
								MicroClusterT model_unk_aux = model_unkT.get(p).get(temp_count_k);
								ArrayList<MicroClusterT> newModels = new ArrayList<MicroClusterT>();

								if ((model_unk_aux != null) && (model_unk_aux.getWeight() >= 0)) {
									if (clusterValidationSilhouetteT(model_unk_aux, p) == true) {

										// ********** The new micro-cluster is valid ****************
										// creating a partial confusion matrix in order to move the unknown examples to
										// the corresponding classes
										mat = new Hashtable();
										// p <- remove
										// O QUE FAZER COM ISSO?????
//										if (p == 0) {
//											for (int count_remove = 0; count_remove < removeExamplesC
//													.get(p).length; count_remove++) {
//												// mark the examples to be removed with the label -2
//												if (removeExamplesC.get(p)[count_remove] == temp_count_k) {
//													removeExamplesC.get(p)[count_remove] = -2;
//
//													if ((mat.get("C "
//															+ dataUnk.get(count_remove)[sizeExample - 1])) == null) {
//														mat.put("C " + dataUnk.get(count_remove)[sizeExample - 1], 1);
//													} else {
//														int valor = ((Integer) mat.get(
//																"C " + dataUnk.get(count_remove)[sizeExample - 1]));
//														mat.put("C " + dataUnk.get(count_remove)[sizeExample - 1],
//																valor + 1);
//													}
//
//												}
//											}
//										}

										// identify the closer micro-cluster to the new valid micro-cluster
										ArrayList<String> ret_func = identifyCloserMicroClusterT(model_unk_aux,
												threshold, p);
										double distance = Double.parseDouble(ret_func.get(0));
										double thresholdValue = Double.parseDouble(ret_func.get(1));
										String category = ret_func.get(2);
										String classLabel = ret_func.get(3);

										// if the distance to the closer micro-cluster is less than a threshold
										String textoArq = "";
										String word = "";
//									if()
										if (distance < thresholdValue) {
											// extension
											if ((category.equalsIgnoreCase("normal"))
													|| (category.equalsIgnoreCase("ext"))) {
												// extension of a class learned offline
												newModels.add(model_unk_aux);
												word = "ext";
												updateModelT(model_unk_aux, classLabel, "ext", p);
												textoArq = "Thinking " + "Extensao: " + "C " + classLabel + " - "
														+ (int) model_unk_aux.getWeight() + " exemplos";

											} else {
												// extension of a novelty pattern
												newModels.add(model_unk_aux);
												word = "extNov";
												updateModelT(model_unk_aux, classLabel, "extNov", p);
												textoArq = "Thinking " + "ExtensaoNovidade: " + "N " + classLabel
														+ " - " + (int) model_unkT.get(p).get(temp_count_k).getWeight()
														+ " exemplos";
											}
										} else {
											// novelty pattern
											newModels.add(model_unk_aux);
											word = "nov";
											updateModelT(model_unk_aux, Integer.toString(noveltyPatterns.size() + 1),
													"nov", p);
											// if the new novelty patter do no intercept any other micro-cluster
											if (modelT.get(p).get(modelT.get(p).size() - 1).getLabelClass()
													.compareToIgnoreCase(
															Integer.toString(noveltyPatterns.size() + 1)) == 0) {
												// if the new valid cluster is a recurrence of known concept
												if (verifyRecurrenceT(model_unk_aux, p)) {
													if (model_unk_aux.getCategory().compareToIgnoreCase("ext") == 0) {
														textoArq = "Thinking " + "Extensao:" + "C "
																+ modelT.get(p).get(modelT.get(p).size() - 1)
																		.getLabelClass()
																+ " - "
																+ (int) model_unkT.get(p).get(temp_count_k).getWeight()
																+ " exemplos";
													} else {
														textoArq = "Thinking " + "ExtensaoNovidade: " + "N "
																+ modelT.get(p).get(modelT.get(p).size() - 1)
																		.getLabelClass()
																+ " - "
																+ (int) model_unkT.get(p).get(temp_count_k).getWeight()
																+ " exemplos";
													}
												} else {
													// novelty pattern was detected
													noveltyPatterns.add(noveltyPatterns.size() + 1 + "");
													textoArq = "ThinkingNov: " + "Novidade " + (noveltyPatterns.size())
															+ " - "
															+ (int) model_unkT.get(p).get(temp_count_k).getWeight()
															+ " exemplos";
												}
											} else {
												textoArq = "Thinking " + "ExtensaoNovidade: " + "N "
														+ modelT.get(p).get(modelT.get(p).size() - 1).getLabelClass()
														+ " - " + (int) model_unkT.get(p).get(temp_count_k).getWeight()
														+ " exemplos";
											}
										}

//									System.out.println(textoArq);
//									System.out.println(mat.values().toArray());
										fileOut.write(textoArq);
										fileOut.write("\n");
										av.addColumnConfusionMatrix(textoArq, mat);
									} // end of valid cluster

								} // end of mininum number of examples

							} // end of each valid cluster
							if (max == model_unkT.get(p).size())
								remove++;
//						}

						}
					}

				}
			}
		}
	}

//	public ArrayList<ArrayList<MicroCluster>> combiner(int numPart, ArrayList<ArrayList<MicroCluster>> model_unkG) {
//
//		for (int p = 0; p < (numPart - 1); p++) {
//			for (int c = 0; c < model_unkG.get(p).size(); c++) {
//
//				double distance = Double.MAX_VALUE;
//				for (int c2 = 0; c2 < model_unkG.get(p + 1).size(); c2++) {
////					System.out.println("?");
//
//					distance = KMeansMOAModified.distance(model_unkG.get(p).get(c).getCenter(),
//							model_unkG.get(p + 1).get(c2).getCenter());
////					
//					double vthreshold = model_unkG.get(p).get(c).getRadius()
//							+ model_unkG.get(p + 1).get(c2).getRadius();
////					
//					String categoria = model_unkG.get(p).get(c).getCategory();
////			
//					if (distance <= vthreshold) {
//
//						model_unkG.get(p + 1).get(c2).setCategory(categoria);
//						model_unkG.get(p + 1).get(c2).setLabelClass("true");
//						model_unkG.get(p).get(c).setLabelClass("true");
//					}
//
//				}
//			}
//		}
//
//		for (int p = 0; p < (numPart - 1); p++) {
//			for (int c = 0; c < model_unkG.get(p).size(); c++) {
//				if (model_unkG.get(p).get(c).getLabelClass().equals("")) {
//					model_unkG.get(p).remove(c);
//				}
//			}
//		}
//
//		return model_unkG;
//
//	}

	// *******************************************************************************
	// ******************** funcao criaModeloDenStream
	// *******************************
	// *******************************************************************************/
	public ArrayList<MicroClusterD> createModelDenStream(String filename, String cl, int[] exampleCluster,
			double epsilon, double mu, double beta, int minPoints) throws NumberFormatException, IOException {
		ArrayList<MicroClusterD> conjModelos = new ArrayList<MicroClusterD>();
		DenStreamOffline jc = new DenStreamOffline();

		Clustering micros = jc.DenStream(cl, filename, epsilon, mu, beta, minPoints);

		if (micros.size() > 0) {
			if (currentState.compareTo("online") == 0) {
				int gruposTemp[] = jc.getClusterExamples();
				for (int i = 0; i < gruposTemp.length; i++) {
					exampleCluster[i] = gruposTemp[i];
				}
			}
		}

		for (int w = 0; w < micros.size(); w++) {
			MicroClusterD mdtemp = new MicroClusterD((MicroClusterMOA) micros.get(w), cl, "normal", timestamp);
			// add the temporary model to the decision model
			conjModelos.add(mdtemp);
		}
		return conjModelos;
	}

	public ArrayList<MicroClusterD> createModelFromExamplesD(ArrayList<String[]> par_data, int par_k, int[] grupos)
			throws IOException {
		ArrayList<MicroClusterD> modelUnk;

		String nomeArqTemp = "caminho_D" + "temp.csv";
		try (FileWriter arquivosSaida = new FileWriter(nomeArqTemp)) {
			for (int i = 0; i < par_data.size(); i++) {
				for (int j = 0; j < par_data.get(i).length - 1; j++) {
					arquivosSaida.write(par_data.get(i)[j] + ",");
				}
				arquivosSaida.write(par_data.get(i)[par_data.get(i).length - 1]);
				arquivosSaida.write("\n");
			}
		}
		
		modelUnk = createModelDenStream(nomeArqTemp, "", grupos, epsilon, mu, beta, minPoints);
		System.out.println("denCluster " + modelUnk.size() );
		
		return (modelUnk);
	}

	// *******************************************************************************
	// ******************** funcao
	// criaModeloCluStree*********************************
	// *******************************************************************************/
	public ArrayList<MicroClusterT> createModelTree(String filename, int alturArvore, String cl, int[] exampleCluster)
			throws NumberFormatException, IOException {
		ArrayList<MicroClusterT> conjModelos = new ArrayList<>();
		ClustreeOffline jc = new ClustreeOffline();

		Clustering micros = jc.TreeStream(cl, filename, alturArvore);

		for (int w = 0; w < micros.size(); w++) {
			MicroClusterT mdtemp = new MicroClusterT((ClusKernel) micros.get(w), cl, "normal", timestamp);
			// add the temporary model to the decision model
			if (mdtemp.getTotalN() > 3) {
				conjModelos.add(mdtemp);
			}
		}
		return conjModelos;
	}

//*******************************************************************************   
//******************** online ***************************************************
//********************************************************************************     
	public void online(Evaluator av, ArrayList<Integer> numPart) throws IOException {
		String classeEx;

		ArrayList<String> noveltyPatterns = new ArrayList<>();
		String filenameOut = outputDirectory + "\\results";
		av.initializeConfusionMatrix(filenameOut);

		String[] data;
		BufferedReader fileOnline = new BufferedReader(new FileReader(new File(filenameOnline)));
		String line;
		int sizeExample = 0;

		// read the online file
		while ((line = fileOnline.readLine()) != null) {
			timestamp++;
			data = line.split(",");
			sizeExample = data.length;
			classeEx = data[sizeExample - 1];

			if ((timestamp % 10000) == 0) {
				System.out.println(timestamp);
			}

			// information about the classification
			String information = "Ex: " + timestamp + "\t Classe Real: " + classeEx + "\t Classe MINAS: ";

			// ************* classify the instance
			// *********************************************
			classify(av, data, information, noveltyPatterns, numPart);

			// put outdated micro-clusters in the sleep memory
			// remove outdated examples from the unknown memory
			if (timestamp % windowSize == 0) {
//				putClusterMemorySleep();
//				removeDataUnk();
			}
		} // # end of the online phase
		fileOnline.close();
		addInfo.close();
		fileOut.close();
		av.showResults(filenameOut);
	}

	// **************************************************************************
	// ******************** updateModelC ****************************************
	// **************************************************************************
	public void updateModelC(MicroClusterC modelUnkC, String classLabel, String category, int num) {
		// update the decision model by adding new valid micro-clusters
		modelUnkC.setCategory(category);
		modelUnkC.setLabelClass(classLabel);
		modelC.get(num).add(modelUnkC);
	}

	// **************************************************************************
	// ******************** updateModelD ****************************************
	// **************************************************************************
	public void updateModelD(MicroClusterD modelUnkD, String classLabel, String category, int num) {
		// update the decision model by adding new valid micro-clusters
		modelUnkD.setCategory(category);
		modelUnkD.setLabelClass(classLabel);
		modelD.get(num).add(modelUnkD);
	}

	// **************************************************************************
	// ******************** updateModel *****************************************
	// **************************************************************************
	public void updateModelT(MicroClusterT modelUnkT, String classLabel, String category, int num) {
		// update the decision model by adding new valid micro-clusters
		modelUnkT.setCategory(category);
		modelUnkT.setLabelClass(classLabel);
		modelT.get(num).add(modelUnkT);
	}

	// **************************************************************************
	// ******************** clusterValidationSilhouette *************************
	// **************************************************************************
	public boolean clusterValidationSilhouette(MicroClusterC modelUnk, int numPart) {
		double minDistance = 0;
		minDistance = KMeansMOAModified.distance(modelC.get(numPart).get(0).getCenter(), modelUnk.getCenter());
		double distance = 0;
		// calculate the distance between the center of the new cluster to the existing
		// clusters
		for (int i = 1; i < modelC.get(numPart).size(); i++) {
			distance = KMeansMOAModified.distance(modelC.get(numPart).get(i).getCenter(), modelUnk.getCenter());
			if (distance < minDistance) {
				minDistance = distance;
			}
		}
		double silhouette = (minDistance - modelUnk.getRadius() / 2) / Math.max(minDistance, modelUnk.getRadius() / 2);
		if (silhouette > 0) {
			return true;
		} else {
			return false;
		}
	}
	
	
	//**************************************************************************   
    //******************** clusterValidationSilhouette *************************
    //**************************************************************************   
    public boolean clusterValidationSilhouetteD(MicroClusterD modelUnk, int numPart) {
        double minDistance;
        minDistance = KMeansMOAModified.distance(modelD.get(numPart).get(0).getCenter(), modelUnk.getCenter());
        double distance;
        // calculate the distance between the center of the new cluster to the existing clusters 
        for (int i = 1; i < modelD.get(numPart).size(); i++) {
            distance = KMeansMOAModified.distance(modelD.get(numPart).get(i).getCenter(), modelUnk.getCenter());
            if (distance < minDistance) {
                minDistance = distance;
            }
        }
        double silhouette = (minDistance - modelUnk.getRadius() / 2) / Math.max(minDistance, modelUnk.getRadius() / 2);
        if (silhouette > 0) {
            return true;
        } else {
            return true;
        }
    }
    
    public boolean clusterValidationSilhouetteT(MicroClusterT modelUnk, int numPart) {
        double minDistance;
        minDistance = KMeansMOAModified.distance(modelT.get(numPart).get(0).getCenter(), modelUnk.getCenter());
        double distance;
        // calculate the distance between the center of the new cluster to the existing clusters 
        for (int i = 1; i < modelT.get(numPart).size(); i++) {
            distance = KMeansMOAModified.distance(modelT.get(numPart).get(i).getCenter(), modelUnk.getCenter());
            if (distance < minDistance) {
                minDistance = distance;
            }
        }
        double silhouette = (minDistance - modelUnk.getRadius() / 2) / Math.max(minDistance, modelUnk.getRadius() / 2);
        if (silhouette > 0) {
            return true;
        } else {
            return true;
        }
    }
    

//*****************************************************************************   
//******************** identifyDistanceCloser *********************************
//*****************************************************************************   
	public ArrayList<String> identifyCloserMicroClusterC(MicroClusterC modelUnk, double threshold, int numPart) {
		double minDistance = 0;
		int posMinDistance = 0;
		minDistance = KMeansMOAModified.distance(modelC.get(numPart).get(0).getCenter(), modelUnk.getCenter());
		double distance = 0;
		// calculate the distance between the center of the new cluster to the existing
		// clusters
		for (int i = 1; i < modelC.get(numPart).size(); i++) {
			distance = KMeansMOAModified.distance(modelC.get(numPart).get(i).getCenter(), modelUnk.getCenter());
			if (distance < minDistance) {
				minDistance = distance;
				posMinDistance = i;
			}
		}
		double vthreshold = modelC.get(numPart).get(posMinDistance).getRadius() / 2 * threshold;

		String categoria = modelC.get(numPart).get(posMinDistance).getCategory();
		String classe = modelC.get(numPart).get(posMinDistance).getLabelClass();

		ArrayList<String> result = new ArrayList<>();
		if (minDistance < vthreshold) {
			result.add(Double.toString(minDistance));
			result.add(Double.toString(vthreshold));
			result.add(categoria);
			result.add(classe);
		} else { // interceptam
			result.add(Double.toString(minDistance));
			vthreshold = modelC.get(numPart).get(posMinDistance).getRadius() / 2 + modelUnk.getRadius() / 2;
			result.add(Double.toString(vthreshold));
			result.add(categoria);
			result.add(classe);
		}
		return result;
	}

	public ArrayList<String> identifyCloserMicroClusterD(MicroClusterD modelUnk, double threshold, int numPart) {
		double minDistance = 0;
		int posMinDistance = 0;
		minDistance = KMeansMOAModified.distance(modelD.get(numPart).get(0).getCenter(), modelUnk.getCenter());
		double distance = 0;
		// calculate the distance between the center of the new cluster to the existing
		// clusters
		for (int i = 1; i < modelD.get(numPart).size(); i++) {
			distance = KMeansMOAModified.distance(modelD.get(numPart).get(i).getCenter(), modelUnk.getCenter());
			if (distance < minDistance) {
				minDistance = distance;
				posMinDistance = i;
			}
		}
		double vthreshold = modelD.get(numPart).get(posMinDistance).getRadius() / 2 * threshold;

		String categoria = modelD.get(numPart).get(posMinDistance).getCategory();
		String classe = modelD.get(numPart).get(posMinDistance).getLabelClass();

		ArrayList<String> result = new ArrayList<>();
		if (minDistance < vthreshold) {
			result.add(Double.toString(minDistance));
			result.add(Double.toString(vthreshold));
			result.add(categoria);
			result.add(classe);
		} else { // interceptam
			result.add(Double.toString(minDistance));
			vthreshold = modelD.get(numPart).get(posMinDistance).getRadius() / 2 + modelUnk.getRadius() / 2;
			result.add(Double.toString(vthreshold));
			result.add(categoria);
			result.add(classe);
		}
		return result;
	}

	public ArrayList<String> identifyCloserMicroClusterT(MicroClusterT modelUnk, double threshold, int numPart) {
		double minDistance = 0;
		int posMinDistance = 0;
		minDistance = KMeansMOAModified.distance(modelT.get(numPart).get(0).getCenter(), modelUnk.getCenter());
		double distance = 0;
		// calculate the distance between the center of the new cluster to the existing
		// clusters
		for (int i = 1; i < modelT.get(numPart).size(); i++) {
			distance = KMeansMOAModified.distance(modelT.get(numPart).get(i).getCenter(), modelUnk.getCenter());
			if (distance < minDistance) {
				minDistance = distance;
				posMinDistance = i;
			}
		}
		double vthreshold = modelT.get(numPart).get(posMinDistance).getRadius() / 2 * threshold;

		String categoria = modelT.get(numPart).get(posMinDistance).getCategory();
		String classe = modelT.get(numPart).get(posMinDistance).getLabelClass();

		ArrayList<String> result = new ArrayList<>();
		if (minDistance < vthreshold) {
			result.add(Double.toString(minDistance));
			result.add(Double.toString(vthreshold));
			result.add(categoria);
			result.add(classe);
		} else { // interceptam
			result.add(Double.toString(minDistance));
			vthreshold = modelT.get(numPart).get(posMinDistance).getRadius() / 2 + modelUnk.getRadius() / 2;
			result.add(Double.toString(vthreshold));
			result.add(categoria);
			result.add(classe);
		}
		return result;
	}

//*******************************************************************************   
//******************** createModelFromExamples **********************************
//*******************************************************************************    
	public ArrayList<MicroClusterC> createModelFromExamples(ArrayList<String[]> par_data, String par_clalg, int par_k,
			int[] grupos) throws IOException {
		ArrayList<MicroClusterC> modelUnk = null;

		if (par_clalg.equals("clustream")) {
			String nomeArqTemp = "caminho" + "temp.csv";
			FileWriter arquivosSaida = new FileWriter(nomeArqTemp);
			for (int i = 0; i < par_data.size(); i++) {
				for (int j = 0; j < par_data.get(i).length - 1; j++) {
					arquivosSaida.write(par_data.get(i)[j] + ",");
				}
				arquivosSaida.write(par_data.get(i)[par_data.get(i).length - 1]);
				arquivosSaida.write("\n");
			}
			arquivosSaida.close();
			modelUnk = criamodeloCluStream(nomeArqTemp, par_k, "", grupos); // grupos, array com 0 de tamanho 2000
		}
		return (modelUnk);
	}

	// *******************************************************************************
	// ******************** createModelFromExamples
	// **********************************
	// *******************************************************************************
	public ArrayList<MicroClusterT> createModelFromExamplesT(ArrayList<String[]> par_data, int alturArvore,
			int[] grupos) throws IOException {
		ArrayList<MicroClusterT> modelUnk = null;

		String nomeArqTemp = "caminho_T" + "temp.csv";
		try (FileWriter arquivosSaida = new FileWriter(nomeArqTemp)) {
			for (int i = 0; i < par_data.size(); i++) {
				for (int j = 0; j < par_data.get(i).length - 1; j++) {
					arquivosSaida.write(par_data.get(i)[j] + ",");
				}
				arquivosSaida.write(par_data.get(i)[par_data.get(i).length - 1]);
				arquivosSaida.write("\n");
			}
			arquivosSaida.close();
		}
		modelUnk = createModelTree(nomeArqTemp, alturArvore, "", grupos); // grupos, array com 0 de tamanho 2000
		System.out.println("TreeCluster " + modelUnk.size() );

		return (modelUnk);
	}

	// *******************************************************************************
	// ******************** funcao criaModeloCluStream
	// **************************************
	// ********************************************************************************/
	public ArrayList<MicroClusterC> criamodeloCluStream(String filename, int numMClusters, String cl,
			int[] exampleCluster) throws NumberFormatException, IOException {
		ArrayList<MicroClusterC> conjModelos = new ArrayList<>();
		ClustreamOffline jc = new ClustreamOffline();

		// execute the clustream with kmeans on offline phase
		boolean executeKMeans = true;
		if (currentState.compareTo("online") == 0) {
			executeKMeans = false; // if online phase, don't execute kmeans
		} else {
			// unica mudança que eu fiz, tirar a execução do kmeans no inicio do algoritmo
			executeKMeans = true; // if offline phase, execute kmeans
		}
		Clustering micros = jc.CluStream(cl, filename, numMClusters, flagMicro, executeKMeans);

		if (micros.size() > 0) {
			if (currentState.compareTo("online") == 0) {
				if (flagMicro) { // so clustream
					int gruposTemp[] = jc.getClusterExamples();
					for (int i = 0; i < gruposTemp.length; i++) {
						exampleCluster[i] = gruposTemp[i];
					}
				} else {// clustrem + kmeans
					int temp[] = jc.getClusteringResults();
					for (int g = 0; g < jc.getClusteringResults().length; g++) {
						exampleCluster[g] = temp[g];
					}
				}
			}
		}

		for (int w = 0; w < micros.size(); w++) {
			MicroClusterC mdtemp = new MicroClusterC((ClustreamKernelMOAModified) micros.get(w), cl, "normal",
					timestamp);
			// add the temporary model to the decision model
			conjModelos.add(mdtemp);
		}

		return conjModelos;
	}

//*******************************************************************************   
//******************** identifyExample ******************************************
//*******************************************************************************    
	public String[] identifyExample(double[] data) throws IOException {
		double distance = 0.0;
		double minDistance = Double.MAX_VALUE;
		int posMinDistance = 0;

		String[] resultC;
		String[] resultD;
		String[] resultT;

		Ensemble ensemble;
		ArrayList<Ensemble> rank = new ArrayList<>();

		// MODEL C
		for (int j = 0; j < modelC.size(); j++) {
			distance = 0.0;
            minDistance = Double.MAX_VALUE;
            posMinDistance = 0;
			
//			System.out.println("modelC.size() " + modelC.size() + " " + modelC.get(j).size());

			resultC = new String[2];

			// for each partition
			for (int k = 0; k < modelC.get(j).size(); k++) {

				distance = KMeansMOAModified.distance(modelC.get(j).get(k).getCenter(), data);
				if (distance <= minDistance) {
					minDistance = distance;
					posMinDistance = k;
				}
			}
			try {
				double raio = modelC.get(j).get(posMinDistance).getRadius();

				if (minDistance < raio) {
					resultC[0] = modelC.get(j).get(posMinDistance).getLabelClass();
					resultC[1] = modelC.get(j).get(posMinDistance).getCategory();
					modelC.get(j).get(posMinDistance).setTime(timestamp);

				}

			} catch (Exception E) {
				System.out.println("pos" + posMinDistance);
				System.out.println("tam modelo: " + modelC.size() + " and : " + j);

			}
//			System.out.println();

			// choice of the partition
			int not = 0;
			if (rank.isEmpty()) {
				ensemble = new Ensemble(resultC, 1);
				rank.add(ensemble);
				addInfo.write(timestamp + "---------------------------------------------\n");
				addInfo.write("Clustream opinion 0 - " + ensemble.getNome(0) + "\n");
				addInfo.write("Clustream opinion 0 - " + ensemble.getNome(1) + "\n");

			} else {
				for (int l = 0; l < rank.size(); l++) {
//					System.out.println("resultC[1] " + resultC[1]);
//					System.out.println("rank.get(l).getNome(1)) " + rank.get(l).getNome(1));
					if (Objects.equals(resultC[1], rank.get(l).getNome(1))) {
						rank.get(l).setQtd(rank.get(l).getQtd() + 1);

						addInfo.write("Clustream opinion " + l + " - " + rank.get(l).getNome(0) + "\n");
						addInfo.write("Clustream opinion " + l + " - " + rank.get(l).getNome(1) + "\n");

						not = 1;
						break;
					}
				}
				if (not == 0) {
					ensemble = new Ensemble(resultC, 1);
					rank.add(ensemble);
				}
			}
		}

		// MODEL D
		for (int j = 0; j < modelD.size(); j++) {
//			System.out.println("modelD.size() " + modelD.size() + " " + modelD.get(j).size());
			resultD = new String[2];
			distance = 0.0;
            minDistance = Double.MAX_VALUE;
            posMinDistance = 0;
			// for each partition
			for (int k = 0; k < modelD.get(j).size(); k++) {
				distance = KMeansMOAModified.distance(modelD.get(j).get(k).getCenter(), data);
				if (distance <= minDistance) {
					minDistance = distance;
					posMinDistance = k;
				}
			}
			try {
				double raio = modelD.get(j).get(posMinDistance).getRadius();

				if (minDistance <= raio) {
					resultD[0] = modelD.get(j).get(posMinDistance).getLabelClass();
					resultD[1] = modelD.get(j).get(posMinDistance).getCategory();
					modelD.get(j).get(posMinDistance).setTime(timestamp);

				}

			} catch (Exception E) {
				System.out.println("pos" + posMinDistance);
				System.out.println("tam modelo: " + modelD.size() + " and : " + j);

			}
			// choice of the partition
			int not = 0;
			if (rank.isEmpty()) {
				ensemble = new Ensemble(resultD, 1);
				rank.add(ensemble);
				addInfo.write(timestamp + "---------------------------------------------\n");
				addInfo.write("DENSTREAM opinion 0 - " + ensemble.getNome(0) + "\n");
				addInfo.write("DENSTREAM opinion 0 - " + ensemble.getNome(1) + "\n");
//				System.out.println("resultD[1] " + resultD[1]);
//				System.out.println("rank.get(l).getNome(1)) " + rank.get(0).getNome(1));

			} else {
				for (int l = 0; l < rank.size(); l++) {
//					System.out.println("resultC[1] " + resultD[1]);
//					System.out.println("rank.get(l).getNome(1)) " + rank.get(l).getNome(1));
					if (Objects.equals(resultD[1], rank.get(l).getNome(1))) {
						rank.get(l).setQtd(rank.get(l).getQtd() + 1);

						addInfo.write("DENSTREAM opinion " + l + " - " + rank.get(l).getNome(0) + "\n");
						addInfo.write("DENSTREAM opinion " + l + " - " + rank.get(l).getNome(1) + "\n");

						not = 1;
						break;
					}
				}
				if (not == 0) {
					ensemble = new Ensemble(resultD, 1);
					rank.add(ensemble);
				}
			}
		}

		// MODEL T

		for (int j = 0; j < modelT.size(); j++) {
//			System.out.println("modelT.size() " + modelT.size());
			resultT = new String[2];
			distance = 0.0;
            minDistance = Double.MAX_VALUE;
            posMinDistance = 0;

			// for each partition
			for (int k = 0; k < modelT.get(j).size(); k++) {

				distance = KMeansMOAModified.distance(modelT.get(j).get(k).getCenter(), data);
				if (distance <= minDistance) {
					minDistance = distance;
					posMinDistance = k;
				}
			}
			try {
				double raio = modelT.get(j).get(posMinDistance).getRadius();

				if (minDistance < raio) {
					resultT[0] = modelT.get(j).get(posMinDistance).getLabelClass();
					resultT[1] = modelT.get(j).get(posMinDistance).getCategory();
					modelT.get(j).get(posMinDistance).setTime(timestamp);

				}

			} catch (Exception E) {
				System.out.println("pos" + posMinDistance);
				System.out.println("tam modelo: " + modelC.size() + " and : " + j);

			}
			// choice of the partition
			int not = 0;
			if (rank.isEmpty()) {
				ensemble = new Ensemble(resultT, 1);
				rank.add(ensemble);
				addInfo.write(timestamp + "---------------------------------------------\n");
				addInfo.write("CLUSTREE opinion 0 - " + ensemble.getNome(0) + "\n");
				addInfo.write("CLUSTREE opinion 0 - " + ensemble.getNome(1) + "\n");
//				System.out.println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
//				System.out.println("resultC[1] " + resultT[1]);
//				System.out.println("rank.get(l).getNome(1)) " + rank.get(0).getNome(1));

			} else {
				for (int l = 0; l < rank.size(); l++) {
//					System.out.println("resultC[1] " + resultT[1]);
//					System.out.println("rank.get(l).getNome(1)) " + rank.get(l).getNome(1));
					if (Objects.equals(resultT[1], rank.get(l).getNome(1))) {
						rank.get(l).setQtd(rank.get(l).getQtd() + 1);

						addInfo.write("CLUSTREE opinion " + l + " - " + rank.get(l).getNome(0) + "\n");
						addInfo.write("CLUSTREE opinion " + l + " - " + rank.get(l).getNome(1) + "\n");

						not = 1;
						break;
					}
				}
				if (not == 0) {
					ensemble = new Ensemble(resultT, 1);
					rank.add(ensemble);
				}
			}
		}

		int maxQtd = Integer.MIN_VALUE;
		int posicao = 0;
		int second = 0;
//		System.out.println("******************************");
//		System.out.println("rank.size " + rank.size());

		for (int i = 0; i < rank.size(); i++) {
			if (maxQtd < rank.get(i).getQtd()) {
				second = posicao;
				posicao = i;
				maxQtd = rank.get(i).getQtd();
			}
			else if (rank.get(i).getQtd() > second && rank.get(i).getQtd() != posicao) 
	            second = rank.get(i).getQtd(); 
			
//			System.out.println(" i " + i + " " + rank.get(i).getNome(0) );
//			System.out.println(" i " + i + " " + rank.get(i).getNome(1) );
//			System.out.println(" i " + i + " " + rank.get(i).getQtd() );

		}
//		System.out.println("maxQtd " + maxQtd);
		if (rank.get(posicao).getNomes()[1] == null) {
			if(second != posicao)
				return rank.get(second).getNomes();
			return null;
		}
		return rank.get(posicao).getNomes();

	}

//*******************************************************************************   
//******************** removeDataUnk ********************************************
//******************************************************************************* 
	public void removeDataUnk() {
		for (int i = 0; i < timestamp_data_unk.size(); i++) {
			if (timestamp_data_unk.get(i) < timestamp - windowSize) {
//              System.out.println("REMOVEU");
				dataUnk.remove(i);
				timestamp_data_unk.remove(i);
			}
		}
	}

//*******************************************************************************   
//******************** putClusterMemorySleep ************************************
//******************************************************************************* 
	public void putClusterMemorySleep() {
		if (alg_C.compareTo("clustream") == 0) {
			for (int i = 0; i < modelC.size(); i++) {
				for (int j = 0; j < modelC.get(i).size(); j++) {
					if (modelC.get(i).get(j).getTime() < (timestamp - windowSize)) {
						SleepModeMemC.get(i).add(modelC.get(i).get(j));
						modelC.get(i).remove(j);
						j--;
					}
				}
			}
		}
		if (alg_D.compareTo("denstream") == 0) {
			for (int i = 0; i < modelD.size(); i++) {
				for (int j = 0; j < modelD.get(i).size(); j++) {
					if (modelD.get(i).get(j).getTime() < (timestamp - windowSize)) {
						SleepModeMemD.get(i).add(modelD.get(i).get(j));
						modelD.get(i).remove(j);
						j--;
					}
				}
			}
		}
		if (alg_T.compareTo("clustree") == 0) {
			for (int i = 0; i < modelT.size(); i++) {
				for (int j = 0; j < modelT.get(i).size(); j++) {
					if (modelT.get(i).get(j).getTime() < (timestamp - windowSize)) {
						SleepModeMemT.get(i).add(modelT.get(i).get(j));
						modelT.get(i).remove(j);
						j--;
					}
				}
			}
		}
	}

//*******************************************************************************   
//******************** funcao verificaReocorrenverifyRecurrence *****************
//******************************************************************************** 
	public boolean verifyRecurrenceC(MicroClusterC modelUnk, int numPart) {
		double distance, minDistance;
		if (SleepModeMemC.get(numPart).size() > 0) {
			minDistance = KMeansMOAModified.distance(SleepModeMemC.get(numPart).get(0).getCenter(),
					modelUnk.getCenter());
			int posMinDistance = 0;

			for (int i = 1; i < SleepModeMemC.size(); i++) {
				distance = KMeansMOAModified.distance(SleepModeMemC.get(numPart).get(i).getCenter(),
						modelUnk.getCenter());
				if (distance < minDistance) {
					minDistance = distance;
					posMinDistance = i;
				}
			}
			double vthreshold = SleepModeMemC.get(posMinDistance).get(numPart).getRadius() / 2 * threshold;

			if ((minDistance) <= vthreshold) {
				if ((SleepModeMemC.get(posMinDistance).get(numPart).getCategory().equalsIgnoreCase("normal"))
						|| (SleepModeMemC.get(numPart).get(posMinDistance).getCategory().equalsIgnoreCase("ext"))) {
					modelUnk.setCategory("ext");
				} else {
					modelUnk.setCategory("extNov");
				}
				modelUnk.setLabelClass(SleepModeMemC.get(numPart).get(posMinDistance).getLabelClass());
				// move micro-cluster from the sleep memory to the decision model
				modelC.add(SleepModeMemC.get(posMinDistance));
				SleepModeMemC.remove(posMinDistance);
				return true;
			}
			return false;
		}
		return false;
	}

	// *******************************************************************************
	// ******************** funcao verificaReocorrenverifyRecurrence
	// *****************
	// ********************************************************************************
	public boolean verifyRecurrenceD(MicroClusterD modelUnk, int numPart) {
		double distance, minDistance;
		if (SleepModeMemD.get(numPart).size() > 0) {
			minDistance = KMeansMOAModified.distance(SleepModeMemD.get(numPart).get(0).getCenter(),
					modelUnk.getCenter());
			int posMinDistance = 0;

			for (int i = 1; i < SleepModeMemD.size(); i++) {
				distance = KMeansMOAModified.distance(SleepModeMemD.get(numPart).get(i).getCenter(),
						modelUnk.getCenter());
				if (distance < minDistance) {
					minDistance = distance;
					posMinDistance = i;
				}
			}
			double vthreshold = SleepModeMemD.get(posMinDistance).get(numPart).getRadius() / 2 * threshold;

			if ((minDistance) <= vthreshold) {
				if ((SleepModeMemD.get(posMinDistance).get(numPart).getCategory().equalsIgnoreCase("normal"))
						|| (SleepModeMemD.get(numPart).get(posMinDistance).getCategory().equalsIgnoreCase("ext"))) {
					modelUnk.setCategory("ext");
				} else {
					modelUnk.setCategory("extNov");
				}
				modelUnk.setLabelClass(SleepModeMemD.get(numPart).get(posMinDistance).getLabelClass());
				// move micro-cluster from the sleep memory to the decision model
				modelD.add(SleepModeMemD.get(posMinDistance));
				SleepModeMemD.remove(posMinDistance);
				return true;
			}
			return false;
		}
		return false;
	}

	// *******************************************************************************
	// ******************** funcao verificaReocorrenverifyRecurrence
	// *****************
	// ********************************************************************************
	public boolean verifyRecurrenceT(MicroClusterT modelUnk, int numPart) {
		double distance, minDistance;
		if (SleepModeMemT.get(numPart).size() > 0) {
			minDistance = KMeansMOAModified.distance(SleepModeMemT.get(numPart).get(0).getCenter(),
					modelUnk.getCenter());
			int posMinDistance = 0;

			for (int i = 1; i < SleepModeMemT.size(); i++) {
				distance = KMeansMOAModified.distance(SleepModeMemT.get(numPart).get(i).getCenter(),
						modelUnk.getCenter());
				if (distance < minDistance) {
					minDistance = distance;
					posMinDistance = i;
				}
			}
			double vthreshold = SleepModeMemT.get(posMinDistance).get(numPart).getRadius() / 2 * threshold;

			if ((minDistance) <= vthreshold) {
				if ((SleepModeMemT.get(posMinDistance).get(numPart).getCategory().equalsIgnoreCase("normal"))
						|| (SleepModeMemT.get(numPart).get(posMinDistance).getCategory().equalsIgnoreCase("ext"))) {
					modelUnk.setCategory("ext");
				} else {
					modelUnk.setCategory("extNov");
				}
				modelUnk.setLabelClass(SleepModeMemT.get(numPart).get(posMinDistance).getLabelClass());
				// move micro-cluster from the sleep memory to the decision model
				modelT.add(SleepModeMemT.get(posMinDistance));
				SleepModeMemT.remove(posMinDistance);
				return true;
			}
			return false;
		}
		return false;
	}

//*******************************************************************************   
//******************** execution ************************************************
//*******************************************************************************
	public void execution() throws IOException {

		Evaluator evaluation = new Evaluator(evaluationType);
		createFileOut();
		System.out.println("\n Results in: " + outputDirectory + "\n\n");

		numPart = new ArrayList<Integer>();
		numPart.add(numC);
		numPart.add(numD);
		numPart.add(numT);

		timestamp++;

		sleepC = new ArrayList<MicroClusterC>();
		sleepD = new ArrayList<MicroClusterD>();
		sleepT = new ArrayList<MicroClusterT>();

		SleepModeMemC = new ArrayList<ArrayList<MicroClusterC>>();
		for (int i = 0; i < numC; i++)
			SleepModeMemC.add(sleepC);
		
		SleepModeMemD = new ArrayList<ArrayList<MicroClusterD>>();
		for (int i = 0; i < numD; i++)
			SleepModeMemD.add(sleepD);
		
		SleepModeMemT = new ArrayList<ArrayList<MicroClusterT>>();
		for (int i = 0; i < numT; i++)
			SleepModeMemT.add(sleepT);

		dataUnk = new ArrayList<>();
		timestamp_data_unk = new ArrayList<>();
		System.out.println("********************** Offline phase (training) ****************************");
		currentState = "offline";
		offline(numPart);
		System.out.println("********************** Online phase (test) ****************************");
		currentState = "online";
		online(evaluation, numPart);
	}

	// *******************************************************************************
	// ******************** main
	// *****************************************************
	// *******************************************************************************
	public static void main(String[] args) throws IOException {
		
		
		String off = "/home/kemilly/euler/MOA3_off";
		String onl = "/home/kemilly/euler/MOA3_onl";
		String result = "resultado/moa_verdadeiro1/";

		int window = 2000;
		int window2 = 4000;
		double thr = 1.1;
		
		int k = 100;
		
		numC = 1;
		numD = 1;
		numT = 1;
		
		if( args.length == 15) {
			System.out.println("ENTROU");
			off = args[0];
			onl = args[1];
			result = args[2];
			
			numC = Integer.parseInt(args[3]);
			numD = Integer.parseInt(args[4]);
			numT = Integer.parseInt(args[5]);
			
			window = Integer.parseInt(args[6]);
			window2 = Integer.parseInt(args[7]);
			
			// clustream
			k = Integer.parseInt(args[8]);				
			thr = Double.parseDouble(args[9])/10;
			
			// denstream
			
			epsilon = Double.parseDouble(args[10])/100;
			mu = Double.parseDouble(args[11]);
			beta = Double.parseDouble(args[12])/10;
			minPoints = Integer.parseInt(args[13]);
		
			// Para o CLUSTREE
			alturArvore = Integer.parseInt(args[14]);
			

		}
				
		System.out.println(off + "\n" + onl + "\n" + result +  "\n" + numC + "\n" + numD +  "\n" + numT + "\n" 
		+ window + "\n" + window2 + "\n" + k + "\n" + thr + "\n" + epsilon + "\n" + mu + "\n" + beta + "\n" +
				minPoints + "\n" + alturArvore);

//		MINAS m = new MINAS("/home/kemilly/euler/MOA3_off", "/home/kemilly/euler/MOA3_onl",
//				"/home/kemilly/resultado/moa2/", "clustream", "denstream", "clustree", 100, 1.1, 2000, 4000, 0, true, 
//				1, 1, 1);

		MINAS m = new MINAS(off, onl, result, k, thr, window, window2, numC, numD, numT, epsilon, mu, beta, 
				minPoints, alturArvore);

//		MINAS m3 = new MINAS("/home/kemi/dev/Dataset-MOA/SynD/SynD_off", "/home/kemi/dev/Dataset-MOA/SynD/SynD_onl",
//				"/home/kemi/dev/Dataset-MOA/resultado/synf/", "clustream", "clustream", 100, 1.1, 2000, 4000, 0, true);
//
//		MINAS m4 = new MINAS("/home/kemi/dev/Dataset-MOA/Kdd99/kdd99_off", "/home/kemi/dev/Dataset-MOA/Kdd99/kdd99_onl",
//				"/home/kemi/dev/Dataset-MOA/resultado/kdd99/", "clustream", "clustream", 100, 1.1, 2000, 4000, 0, true);
//
//		MINAS m5 = new MINAS("/home/kemi/dev/Dataset-MOA/covtypeNorm_off", "/home/kemi/dev/Dataset-MOA/covtypeNorm_onl",
//				"/home/kemi/dev/Dataset-MOA/resultado/cov/", "clustream", "clustream", 100, 1.1, 2000, 4000, 0, true);
//		
//		MINAS m6 = new MINAS("/home/kemilly/cov123_train.txt", "/home/kemilly/cov123_test.txt",
//				"/home/kemilly/Experimentos/Exp2019/covertype/minas/", "clustream", "clustream", 100, 1.1, 2000, 4000, 0, true);

//		m6.execution();
		m.execution();
//		m2.execution();
//		m3.execution();
//		m4.execution();
//		m5.execution();

	}
}
