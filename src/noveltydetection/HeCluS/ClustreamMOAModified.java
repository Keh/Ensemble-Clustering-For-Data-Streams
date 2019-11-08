package noveltydetection.HeCluS;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

import moa.cluster.CFCluster;
import moa.cluster.Cluster;
import moa.cluster.Clustering;
import moa.cluster.SphereCluster;
import moa.clusterers.AbstractClusterer;
import moa.clusterers.KMeans;
import moa.clusterers.streamkm.MTRandom;
import moa.core.Measurement;
import moa.options.IntOption;
import weka.core.DenseInstance;
import weka.core.Instance;

public class ClustreamMOAModified extends AbstractClusterer {
	public IntOption timeWindowOption = new IntOption("timeWindow", 't', "Rang of the window.", 1000);

	public IntOption maxNumKernelsOption = new IntOption("maxNumKernels", 'k',
			"Maximum number of micro kernels to use.", 100);

	public static final int m = 50;
	public static final int t = 2;
	private int timeWindow;
	private long timestamp = -1;
	private ClustreamKernelMOAModified[] kernels;
	private boolean initialized;
	private List<ClustreamKernelMOAModified> buffer; // Buffer for initialization with kNN
	private int bufferSize;
	private ArrayList<ArrayList<Integer>> clusterExamples;

	@Override
	public void resetLearningImpl() {
		this.timeWindow = timeWindowOption.getValue();
		this.initialized = false;
		this.buffer = new LinkedList<ClustreamKernelMOAModified>();
	}

	public ArrayList<ArrayList<Integer>> getClusterExamples() {
		return clusterExamples;
	}

	public void initializeBufferSize(int initNumber, int valor) {
		this.bufferSize = /* valor */initNumber;
		this.kernels = new ClustreamKernelMOAModified[/* this.bufferSize */valor];
		clusterExamples = new ArrayList<ArrayList<Integer>>();
	}

	@Override
	public void trainOnInstanceImpl(Instance instance) {
		int dim = instance.numValues();
		// same number of kernels
		if (timestamp == -1) {
			for (int i = 0; i < this.kernels.length; i++)
				clusterExamples.add(new ArrayList<Integer>());
		}
		timestamp++;
		// 0. Initialize
		if (!initialized) {
			if (buffer.size() < bufferSize) {
				buffer.add(new ClustreamKernelMOAModified(instance, dim, timestamp));
				return;
			}

			int k = kernels.length;
			assert (k < bufferSize);

			ClustreamKernelMOAModified[] centers = new ClustreamKernelMOAModified[k];
			int nroaleatorio;
			List<Integer> numeros = new ArrayList<Integer>();
			for (int cont = 0; cont < buffer.size(); cont++) {
				numeros.add(cont);
			}
			Collections.shuffle(numeros);

			for (int i = 0; i < k; i++) {
				// centers[i] = buffer.get( i ); // TODO: make random!
				nroaleatorio = numeros.get(i).intValue();
				centers[i] = buffer.get(nroaleatorio);
			}

			Clustering kmeans_clustering = kMeans(k, centers, buffer, clusterExamples);
			for (int i = 0; i < kmeans_clustering.size(); i++) {
				kernels[i] = new ClustreamKernelMOAModified(new DenseInstance(1.0, centers[i].getCenter()), dim,
						timestamp);
//				System.out.println(kernels[i].getCenter());
//				System.out.println(kernels[i].getRadius());
			}
			buffer.clear();
			initialized = true;
			// return;

		}

		// 1. Determine closest kernel
		ClustreamKernelMOAModified closestKernel = null;
		double minDistance = Double.MAX_VALUE;
		int kernelProximo = 0;

		for (int i = 0; i < kernels.length; i++) {
			// System.out.println(i+" "+kernels[i].getWeight()+"
			// "+kernels[i].getDeviation());
			double distance = distance(instance.toDoubleArray(), kernels[i].getCenter());
			if (distance < minDistance) {
				closestKernel = kernels[i];
				kernelProximo = i;
				minDistance = distance;
			}
		}
		// 2. Check whether instance fits into closestKernel
		double radius = 0.0;
		if (closestKernel.getWeight() == 1) {
			// Special case: estimate radius by determining the distance to the
			// next closest cluster
			radius = Double.MAX_VALUE;
			double[] center = closestKernel.getCenter();
			for (int i = 0; i < kernels.length; i++) {
				if (kernels[i] == closestKernel) {
					continue;
				}

				double distance = distance(kernels[i].getCenter(), center);
				radius = Math.min(distance, radius);
			}
		} else {
			radius = closestKernel.getRadius();
		}

		if (minDistance < radius) {
			// Date fits, put into kernel and be happy
			// System.out.print(kernelProximo+",");
			closestKernel.insert(instance, timestamp);
			clusterExamples.get(kernelProximo).add((int) timestamp);
			return;
		}

		// 3. Date does not fit, we need to free
		// some space to insert a new kernel
		long threshold = timestamp - timeWindow; // Kernels before this can be forgotten

		// 3.1 Try to forget old kernels
		/*
		 * for ( int i = 0; i < kernels.length; i++ ) { if (
		 * kernels[i].getRelevanceStamp() < threshold ) { kernels[i] = new
		 * ClustreamKernel( instance, dim, timestamp ); grupo.remove(i); grupo.add(i,
		 * new ArrayList<Integer>()); grupo.get(i).add((int)timestamp);
		 * //System.out.println("\n*****aqui"); return; } }
		 */

		// 3.2 Merge closest two kernels
		int closestA = 0;
		int closestB = 0;
		minDistance = Double.MAX_VALUE;
		for (int i = 0; i < kernels.length; i++) {
			double[] centerA = kernels[i].getCenter();
			for (int j = i + 1; j < kernels.length; j++) {
				double dist = distance(centerA, kernels[j].getCenter());
				if (dist < minDistance) {
					minDistance = dist;
					closestA = i;
					closestB = j;
				}
			}
		}
		assert (closestA != closestB);

		kernels[closestA].add(kernels[closestB]);
		kernels[closestB] = new ClustreamKernelMOAModified(instance, dim, timestamp);
		for (int i = 0; i < clusterExamples.get(closestB).size(); i++) {
			clusterExamples.get(closestA).add(clusterExamples.get(closestB).get(i));
		}
		clusterExamples.set(closestB, new ArrayList<Integer>());
		clusterExamples.get(closestB).add((int) timestamp);
		
	}

	@Override
	public Clustering getMicroClusteringResult() {
		if (!initialized) {
			return new Clustering(new Cluster[0]);
		}

		ClustreamKernelMOAModified[] res = new ClustreamKernelMOAModified[kernels.length];
		for (int i = 0; i < res.length; i++) {
			res[i] = new ClustreamKernelMOAModified(kernels[i]);
		}

		return new Clustering(res);
	}

	@Override
	public boolean implementsMicroClusterer() {
		return true;
	}

	@Override
	public Clustering getClusteringResult() {
		return null;
	}

	public String getName() {
		return "Clustream " + timeWindow;
	}

	private static double distance(double[] pointA, double[] pointB) {
		double distance = 0.0;
		for (int i = 0; i < pointA.length; i++) {
			double d = pointA[i] - pointB[i];
			distance += d * d;
		}
		return Math.sqrt(distance);
	}

	public static Clustering kMeans(int k, Cluster[] centers, List<? extends Cluster> data,
			ArrayList<ArrayList<Integer>> grupo) {
		assert (centers.length == k);
		assert (k > 0);

		int dimensions = centers[0].getCenter().length;

		ArrayList<ArrayList<Cluster>> clustering = new ArrayList<ArrayList<Cluster>>();
		for (int i = 0; i < k; i++) {
			clustering.add(new ArrayList<Cluster>());
		}

		int repetitions = 100;
		int m;
		while (repetitions-- >= 0) {
			// Assign points to clusters
//			System.out.println("repetitions "  + repetitions);
			m = 0;
			for (Cluster point : data) {
				double minDistance = distance(point.getCenter(), centers[0].getCenter());
				int closestCluster = 0;
				for (int i = 1; i < k; i++) {
					double distance = distance(point.getCenter(), centers[i].getCenter());
					if (distance < minDistance) {
						closestCluster = i;
						minDistance = distance;
					}
				}

				clustering.get(closestCluster).add(point);
				if (repetitions == 0) {
					grupo.get(closestCluster).add(m);
				}
				m++;
			}

			// Calculate new centers and clear clustering lists
			SphereCluster[] newCenters = new SphereCluster[centers.length];
			for (int i = 0; i < k; i++) {
				newCenters[i] = calculateCenter(clustering.get(i), dimensions);
				clustering.get(i).clear();
			}
			centers = newCenters;
		}
		return new Clustering(centers);
	}

	private static SphereCluster calculateCenter(ArrayList<Cluster> cluster, int dimensions) {
		double[] res = new double[dimensions];
		for (int i = 0; i < res.length; i++) {
			res[i] = 0.0;
		}

		if (cluster.size() == 0) {
			return new SphereCluster(res, 0.0);
		}

		for (Cluster point : cluster) {
			double[] center = point.getCenter();
			for (int i = 0; i < res.length; i++) {
				res[i] += center[i];
			}
		}

		// Normalize
		for (int i = 0; i < res.length; i++) {
			res[i] /= cluster.size();
		}

		// Calculate radius
		double radius = 0.0;
		for (Cluster point : cluster) {
			double dist = distance(res, point.getCenter());
			if (dist > radius) {
				radius = dist;
			}
		}
		SphereCluster sc = new SphereCluster(res, radius);
		sc.setWeight(cluster.size());
		return sc;
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		throw new UnsupportedOperationException("Not supported yet.");
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		throw new UnsupportedOperationException("Not supported yet.");
	}

	public boolean isRandomizable() {
		return false;
	}

	public double[] getVotesForInstance(Instance inst) {
		throw new UnsupportedOperationException("Not supported yet.");
	}

	public ArrayList<ArrayList<Integer>> getGrupo() {
		return clusterExamples;
	}

	public static Cluster[] centrosKMeansMais(int k, int n, int d, ArrayList<CFCluster> points) {

		// array to store the choosen centres
		Cluster[] centres = new Cluster[k];
		MTRandom clustererRandom;
		clustererRandom = new MTRandom(System.currentTimeMillis());
		int centreIndex[] = new int[points.size()];
		double curCost[] = new double[points.size()];

		// choose the first centre (each point has the same probability of being
		// choosen)
		int i = 0;

		int next = 0;
		int j = 0;
		// do{ //only choose from the n-i points not already choosen
		next = clustererRandom.nextInt(n - 1);

		// check if the choosen point is not a dummy
		// } while( points[next].weight < 1);

		// set j to next unchoosen point
		j = next;
		// copy the choosen point to the array
		centres[i] = points.get(j);

		// set the current centre for all points to the choosen centre
		for (i = 0; i < n; i++) {
			centreIndex[i] = 0;
			curCost[i] = distance(points.get(i).getCenter(), centres[0].getCenter());
		}
		// choose centre 1 to k-1 with the kMeans++ distribution
		for (i = 1; i < k; i++) {

			double cost = 0.0;
			for (j = 0; j < n; j++) {
				cost += curCost[j];
			}

			double random = 0;
			double sum = 0.0;
			int pos = -1;

			// do{
			random = clustererRandom.nextDouble();// genrand_real3();
			sum = 0.0;
			pos = -1;

			for (j = 0; j < n; j++) {
				sum = sum + curCost[j];
				if (random <= sum / cost) {
					pos = j;
					break;
				}
			}
			// } while (points[pos].weight < 1);

			// copy the choosen centre
			centres[i] = points.get(pos);
			// check which points are closest to the new centre
			for (j = 0; j < n; j++) {
				double newCost = distance(points.get(j).getCenter(), centres[i].getCenter());
				if (curCost[j] > newCost) {
					curCost[j] = newCost;
					centreIndex[j] = i;
				}
			}

		}

		return centres;
	}

}
