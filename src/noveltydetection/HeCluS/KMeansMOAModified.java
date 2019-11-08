/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package noveltydetection.HeCluS;


import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import moa.cluster.*;
import moa.clusterers.clustream.ClustreamKernel;
import moa.clusterers.streamkm.MTRandom;

/**
 *
 * @author jansen
 */
public class KMeansMOAModified {
	public static int melhorK;
	public static int verdadeiroK;
    public int tam [];
    private ArrayList<ArrayList<Cluster>> agrupamento ;
    private double []distMax;
    private double []withinss;
    private int []clusters;
    private double [] distMedia;
    private String classesGrupos [];

	public Clustering kMeans2(Cluster[] centers, List<? extends Cluster> data,double soma[] ) {
        int k = centers.length;
        tam = new int[k];

        int dimensions = centers[0].getCenter().length;
        clusters = new int [data.size()];

        ArrayList<ArrayList<Cluster>> clustering = new ArrayList<ArrayList<Cluster>>();
        for ( int i = 0; i < k; i++ ) {
        	clustering.add( new ArrayList<Cluster>() );
        }

        int repetitions = 100;
        double sum=0;
        do {
        	// Assign points to clustersf
            int cont =0;
            sum = 0; 
        	for ( Cluster point : data ) { 
        		double minDistance = distance( point.getCenter(), centers[0].getCenter() );
        		int closestCluster = 0;
        		for ( int i = 1; i < k; i++ ) {
        			double distance = distance( point.getCenter(), centers[i].getCenter() );
        			//if ( round(distance,6) < round(minDistance,6)) {
                     if ( distance < minDistance) {
        				closestCluster = i;
        				minDistance = distance;
        			}
        		}
        		sum +=minDistance;
        		clustering.get( closestCluster ).add( point );
                        
                if (repetitions == 0 ){
                        clusters[cont] = closestCluster;                        
                }
                cont++;
        	}
        	// Calculate new centers and clear clustering lists
        	SphereCluster[] newCenters = new SphereCluster[centers.length];
        	for ( int i = 0; i < k; i++ ) {                   
        		newCenters[i] = calculateCenter( clustering.get( i ), dimensions );
                if (repetitions != 0 ){
                     clustering.get( i ).clear();
                }
        	}                
        	centers = newCenters;
            repetitions--;
	    }while ( repetitions >= 0 );
              
        agrupamento = clustering;
        soma[0] = sum;       
        return new Clustering( centers );
    }
    
    public static Clustering kMeansNaoVazio(Cluster[] centers, List<? extends Cluster> data,double soma[] ) {
        int k = centers.length;

        int dimensions = centers[0].getCenter().length;

        ArrayList<ArrayList<Cluster>> clustering = new ArrayList<ArrayList<Cluster>>();
        for ( int i = 0; i < k; i++ ) {
        	clustering.add( new ArrayList<Cluster>() );
        }

        int repetitions = 100;
        double sum=0;
	
		while ( repetitions-- >= 0 ) {
		    // Assign points to clusters
			sum = 0;
		    for ( Cluster point : data ) {
				double minDistance = distance( point.getCenter(), centers[0].getCenter() );
				int closestCluster = 0;
				for ( int i = 1; i < k; i++ ) {
				    double distance = distance( point.getCenter(), centers[i].getCenter() );
				    if ( distance < minDistance ) {
				    	closestCluster = i;
				    	minDistance = distance;
				    }
				}
				sum +=minDistance;
				clustering.get( closestCluster ).add( point );
		    }
	
	    // Calculate new centers and clear clustering lists
	    SphereCluster[] newCenters = new SphereCluster[centers.length];
	    for ( int i = 0; i < k; i++ ) {
    		int posMaior1=0, posMaior2=0;
	    	if (clustering.get(i).size() == 0){ // cluster nao tem elementos
	    		System.out.println("************tamanho zero*************");
	    		//buscar o cluster com maior raio: 
	    		//posMaior1: cluster com o maior raio
	    		//posMaior2: elemento do cluster com maior distancia ao centroide
	    		double maiorraio = 0;
	    		double distance;
	    		for (int j = 0; j<k; j++){
	    			if (i!=j){
	    				for (int l=0; l<clustering.get(j).size();l++){
	    					distance = distance(clustering.get(j).get(l).getCenter(),centers[j].getCenter());
	    					if (distance > maiorraio){
	    						maiorraio = distance;
	    						posMaior1 = j;
	    						posMaior2 = l;
	    					}
	    				}
	    			}
	    		}
	    		//criando um novo grupo com este elemento mais distance
	    		clustering.get(i).add(clustering.get(posMaior1).get(posMaior2));
	    		clustering.get(posMaior1).remove(posMaior2);
	    		
	    		//Redistribuindo os elementos entre os 2 clustrs 
	    		for (int n =0; n< clustering.get(posMaior1).size();n++){
	    			double distance1 = distance(clustering.get(posMaior1).get(n).getCenter(),centers[posMaior1].getCenter());
	    			double distance2 = distance(clustering.get(posMaior1).get(n).getCenter(),clustering.get(i).get(0).getCenter());
	    			if (distance2< distance1){
	    				clustering.get(i).add(clustering.get(posMaior1).get(n));
	    				clustering.get(posMaior1).set(n, null);
	    			}
	    		}
	    		for (int z = 0; z <clustering.get(posMaior1).size(); z++)
	    			if (clustering.get(posMaior1).get(z) == null){
	    				clustering.get(posMaior1).remove(z);
	    				z--;
	    			}
	    		
	    		//recalculando o centro do cluster q foi diminuido
	    		newCenters[posMaior1] = calculateCenter( clustering.get( posMaior1 ), dimensions );
	    	}
	    	newCenters[i] = calculateCenter( clustering.get( i ), dimensions );
	    }
	    centers = newCenters;
	    for (int i=0; i< k; i++)
	    	clustering.get( i ).clear();
	}
	soma[0] = sum;
	return new Clustering( centers );
    }

    public int [] getClusters(){
            return clusters;
    }
        
    public void distanciaMaxima(Clustering clustering){
        int k = agrupamento.size();        
        distMax = new double[k];       
        withinss = new double[k];    
		double [] centro;
		double maiorDist;
		double [] elem;
		double soma, distancia;	
		distMedia = new double[k];
       
		for (int i=0; i<k; i++){ // para cada macro-cluster
            // clustering eh o resultado do kmeans
			centro = clustering.get(i).getCenter(); //obter o centro do macro-cluster            
			maiorDist = 0;
              
			for (int j=0;j<agrupamento.get(i).size(); j++){ //para cada elemento no macro-cluster
				elem = agrupamento.get(i).get(j).getCenter();
				soma=0;
				for (int l=0; l< elem.length; l++){ //distancia do elem ao centro
					soma = soma + Math.pow(centro[l] - elem[l], 2);
				}
				distancia = Math.sqrt(soma);
				distMedia[i] = distMedia[i]+distancia;
				withinss[i] = withinss[i] + (distancia*distancia);
				if (distancia > maiorDist) // procurando pelo elem cuja distancia e a maior
	   				maiorDist = distancia;
				}
	   			distMax[i] = maiorDist;
			}  
		}  
    
    public double [] getTam(){
        double tam[] = new double[agrupamento.size()]; 
               
        for (int i=0;i<tam.length; i++)
            tam[i] = agrupamento.get(i).size();      
        return tam;
    }
    
    public double[] getdistMax(){  
        return distMax;
    }
    
    public double[] getwithinss(){
        return withinss;
    }
    public double[] getdistMedia(){
    	double res[] = new double[agrupamento.size()];
		for (int i=0; i<res.length;i++){
			res[i]= distMedia[i]/agrupamento.get(i).size();
		}
		return res;    	    	
    }
    public String[] getClassesGrupos(){
    	return classesGrupos;
    }
	
    public static Clustering kMeans(Cluster[] centers, List<? extends Cluster> data ) {
	    int k = centers.length;
	
		int dimensions = centers[0].getCenter().length;
	
		ArrayList<ArrayList<Cluster>> clustering =
			new ArrayList<ArrayList<Cluster>>();
		for ( int i = 0; i < k; i++ ) {
		    clustering.add( new ArrayList<Cluster>() );
		}
	
		int repetitions = 100;
		while ( repetitions-- >= 0 ) {
		    // Assign points to clusters
		    for ( Cluster point : data ) {
			double minDistance = distance( point.getCenter(), centers[0].getCenter() );
			int closestCluster = 0;
			for ( int i = 1; i < k; i++ ) {
			    double distance = distance( point.getCenter(), centers[i].getCenter() );
			    if ( distance < minDistance ) {
				closestCluster = i;
				minDistance = distance;
			    }
			}
	
			clustering.get( closestCluster ).add( point );
		    }
	
		    // Calculate new centers and clear clustering lists
		    SphereCluster[] newCenters = new SphereCluster[centers.length];
		    for ( int i = 0; i < k; i++ ) {
			newCenters[i] = calculateCenter( clustering.get( i ), dimensions );
			clustering.get( i ).clear();
		    }
		    centers = newCenters;
		}
	
		return new Clustering( centers );
    }

    public static double round(double d, int decimalPlace){
	    // see the Javadoc about why we use a String in the constructor
	    // http://java.sun.com/j2se/1.5.0/docs/api/java/math/BigDecimal.html#BigDecimal(double)
	    BigDecimal bd = new BigDecimal(Double.toString(d));
	    bd = bd.setScale(decimalPlace,BigDecimal.ROUND_HALF_UP);
	    return bd.doubleValue();
	  }
    
    public static double distance(double[] pointA, double [] pointB){
        double distance = 0.0;
        for (int i = 0; i < pointA.length; i++) {
            //double d = round(pointA[i],7) - round(pointB[i],7);
        	double d = pointA[i] - pointB[i];
            distance += d * d;
        }
        return Math.sqrt(distance);  //distance;         
    }


    private static SphereCluster calculateCenter( ArrayList<Cluster> cluster, int dimensions ) {
	double[] res = new double[dimensions];
	for ( int i = 0; i < res.length; i++ ) {
	    res[i] = 0.0;
	}

	if ( cluster.size() == 0 ) {
	    return new SphereCluster( res, 0.0 );
	}

	for ( Cluster point : cluster ) {
            double [] center = point.getCenter();
            for (int i = 0; i < res.length; i++) {
               res[i] += center[i];
            }
	}

	// Normalize
	for ( int i = 0; i < res.length; i++ ) {
	    res[i] /= cluster.size();
	}

	// Calculate radius
	double radius = 0.0;
	for ( Cluster point : cluster ) {
	    double dist = distance( res, point.getCenter() );
	    if ( dist > radius ) {
		radius = dist;
	    }
	}

	return new SphereCluster( res, radius );
    }

    public static Clustering gaussianMeans(Clustering gtClustering, Clustering clustering) {
        ArrayList<CFCluster> microclusters = new ArrayList<CFCluster>();
        for (int i = 0; i < clustering.size(); i++) {
            if (clustering.get(i) instanceof CFCluster) {
                microclusters.add((CFCluster)clustering.get(i));
            }
            else{
                System.out.println("Unsupported Cluster Type:"+clustering.get(i).getClass()
                        +". Cluster needs to extend moa.cluster.CFCluster");
            }
        }
        Cluster[] centers = new Cluster[gtClustering.size()];
        for (int i = 0; i < centers.length; i++) {
            centers[i] = gtClustering.get(i);

        }
                  
    int k = centers.length;
	if ( microclusters.size() < k ) {
	    return new Clustering( new Cluster[0]);
	}

	Clustering kMeansResult = kMeans( centers, microclusters );

	k = kMeansResult.size();
	CFCluster[] res = new CFCluster[ k ];

	for ( CFCluster microcluster : microclusters) {
	    // Find closest kMeans cluster
	    double minDistance = Double.MAX_VALUE;
	    int closestCluster = 0;
	    for ( int i = 0; i < k; i++ ) {
		double distance = distance( kMeansResult.get(i).getCenter(), microcluster.getCenter() );
		if ( distance < minDistance ) {
		    closestCluster = i;
		    minDistance = distance;
		}
	    }

	    // Add to cluster
	    if ( res[closestCluster] == null ) {
		res[closestCluster] = (CFCluster)microcluster.copy();
	    } else {
		res[closestCluster].add( microcluster );
	    }
	}

	// Clean up res
	int count = 0;
	for ( int i = 0; i < res.length; i++ ) {
	    if ( res[i] != null )
		++count;
	}

	CFCluster[] cleaned = new CFCluster[count];
	count = 0;
	for ( int i = 0; i < res.length; i++ ) {
	    if ( res[i] != null )
		cleaned[count++] = res[i];
	}

	return new Clustering( cleaned );
    }

    
    public static Clustering gaussianMeans2(int nrogrupos, Clustering clustering, ArrayList<ArrayList<Integer>>grupo) {
        ArrayList<CFCluster> microclusters = new ArrayList<CFCluster>();   
        ArrayList<ArrayList<Integer>>grupoAux = new ArrayList<ArrayList<Integer>>();
        
        for (int i = 0; i < clustering.size(); i++) {
            if (clustering.get(i) instanceof CFCluster) {
                microclusters.add((CFCluster)clustering.get(i));
            }
            else{
                System.out.println("Unsupported Cluster Type:"+clustering.get(i).getClass()
                        +". Cluster needs to extend moa.cluster.CFCluster");
            }
        }
        Cluster[] centers = new Cluster[nrogrupos];
        /*for (int i = 0; i < centers.length; i++) {
            centers[i] = clustering.get(i);
        	grupoAux.add(new ArrayList<Integer>());
        }*/
            
        /*int nroaleatorio;
        List<Integer> numeros = new ArrayList<Integer>();  
        for (int cont = 0; cont < clustering.size(); cont++) { 
            numeros.add(cont);  
        }  
        Collections.shuffle(numeros); 
        for (int i = 0; i < centers.length; i++) {
        	nroaleatorio = numeros.get(i).intValue();
            centers[i] = clustering.get(nroaleatorio);
            //centers[i] = clustering.get(i);                     
   			grupo[i] = new ArrayList<Integer>();

        }*/
        //Modo1: centros com probabilidade proporcional ao tamanho dos grupos
        //ordena os microclusters em ordem decrescente de tamanho
         Collections.sort (microclusters, new Comparator() {  
             public int compare(Object o1, Object o2) {  
            	 CFCluster p1 = (CFCluster) o1;  
            	 CFCluster p2 = (CFCluster) o2;  
                 return p1.getWeight() > p2.getWeight() ? -1 : (p1.getWeight() < p2.getWeight() ? +1 : 0);  
             }  
         });  
         int tamanhoTotal=0;
         //calculo do tamanho total de elementos nos microclusters
         for (int i=0; i<microclusters.size(); i++){
         	tamanhoTotal+= microclusters.get(i).getWeight();
         }
         MTRandom m = new MTRandom(System.currentTimeMillis());
         //MTRandom m = new MTRandom(1);
         int nroaleatorio=0, nromicro=0;
         int nrocentros = centers.length;
         ArrayList<Integer>NroSorteados = new ArrayList<Integer>();
         for (int i = 0; i < nrocentros; i++) {
           	grupoAux.add(new ArrayList<Integer>());
         }
         
         for (int i = 0; i < nrocentros; i++) {
         	nroaleatorio = m.nextInt(tamanhoTotal);
         	//busca em qual microclusters esta o nro aleatorio
         	int soma = 0;
         	for (int j=0; j<microclusters.size(); j++){
         		if ((nroaleatorio >= soma) && (nroaleatorio < microclusters.get(j).getWeight()+soma)){
         				if (!(NroSorteados.contains(j))){
         					nromicro = j;
         					NroSorteados.add(nromicro);
         					break;
         				}
         				else{
         					i--;
         					break;
         				}
         		}
         		soma+= (int) microclusters.get(j).getWeight();
         	}        	
         	centers[i] = microclusters.get(nromicro);
         }


        int k = centers.length;
	if ( microclusters.size() < k ) {
	    return new Clustering( new Cluster[0]);
	}

	Clustering kMeansResult = kMeans( centers, microclusters );

	k = kMeansResult.size();
	CFCluster[] res = new CFCluster[ k ];

	int contador = 0;
	for ( CFCluster microcluster : microclusters) {
	    // Find closest kMeans cluster
	    double minDistance = Double.MAX_VALUE;
	    int closestCluster = 0;
	    for ( int i = 0; i < k; i++ ) {
		double distance = distance( kMeansResult.get(i).getCenter(), microcluster.getCenter() );
		if ( distance < minDistance ) {
		    closestCluster = i;
		    minDistance = distance;
		}
	    }

	    // Add to cluster
	    if ( res[closestCluster] == null ) {
		res[closestCluster] = (CFCluster)microcluster.copy();
	    } else {
		res[closestCluster].add( microcluster );
	    }
	    grupoAux.get(closestCluster).add((int)contador);
	    contador++;
	}

	// Clean up res
	int count = 0;
	for ( int i = 0; i < res.length; i++ ) {
	    if ( res[i] != null )
		++count;
	}

	CFCluster[] cleaned = new CFCluster[count];
	count = 0;
	for ( int i = 0; i < res.length; i++ ) {
	    if ( res[i] != null ){
	    	cleaned[count++] = res[i];
	    	grupo.add(grupoAux.get(i));
	    }
	}

	return new Clustering( cleaned );
    }
    
    public static Clustering gaussianMeans3(int nroGrupos, Clustering clustering, int algEscolhaCentros,ArrayList<ArrayList<Integer>>grupo) {
        ArrayList<CFCluster> microclusters = new ArrayList<CFCluster>();
        ArrayList<ArrayList<Integer>>grupoAux = new ArrayList<ArrayList<Integer>>();
        
        for (int i = 0; i < clustering.size(); i++) {
            if (clustering.get(i) instanceof CFCluster) {
                microclusters.add((CFCluster)clustering.get(i));
            }
            else{
                System.out.println("Unsupported Cluster Type:"+clustering.get(i).getClass()
                        +". Cluster needs to extend moa.cluster.CFCluster");
            }
        }
        Cluster[] centers = new Cluster[nroGrupos];
        Cluster[] melhoresCentros = null;

        //feito por Elaine
        for (int i = 0; i < nroGrupos; i++) {
           	grupoAux.add(new ArrayList<Integer>());
         }
        
        double CurCusto=0, MinCusto=0;
        switch (algEscolhaCentros){
			//case 1: melhoresCentros = centrosMOA(microclusters, nroGrupos, gtClustering); break;
			case 2: melhoresCentros = centrosAleatorios(microclusters, nroGrupos); break; 
			case 3: melhoresCentros = centrosKMeansMais(nroGrupos, microclusters.size(),microclusters.get(0).getCenter().length, microclusters); break;
			case 4:melhoresCentros = centrosClustream(microclusters, nroGrupos); break;
			case 0:melhoresCentros = centrosSeq(microclusters, nroGrupos); break;
        }   	    
        centers = melhoresCentros.clone();
        int k = centers.length;
		if ( microclusters.size() < k ) {
		    return new Clustering( new Cluster[0]);
		}
	//feito por Elaine
	double Custo[] = new double[1];
    Clustering kMeansResult = kMeansNaoVazio( centers, microclusters, Custo );    
    CurCusto = Custo[0];
    MinCusto = CurCusto;
    for (int i=1; i<5; i++){
    	switch (algEscolhaCentros){
    		//case 1: melhoresCentros = centrosMOA(microclusters, nroGrupos, gtClustering); break;
    		case 2: melhoresCentros = centrosAleatorios(microclusters, nroGrupos); break; 
    		case 3: melhoresCentros = centrosKMeansMais(nroGrupos, microclusters.size(),microclusters.get(0).getCenter().length, microclusters); break;
    		case 4:melhoresCentros = centrosClustream(microclusters, nroGrupos); break;
    		case 0:melhoresCentros = centrosSeq(microclusters, nroGrupos); break;
    	}    
                      
    	kMeansResult = kMeansNaoVazio( melhoresCentros, microclusters, Custo );
    	CurCusto = Custo[0];
    	if(CurCusto < MinCusto) {
			MinCusto = CurCusto;
			//centers = melhoresCentros;
			for (int m = 0; m<centers.length; m++)
				centers[m] = melhoresCentros[m];
		}
    }

	kMeansResult = kMeansNaoVazio( centers, microclusters ,Custo);

	k = kMeansResult.size();
	CFCluster[] res = new CFCluster[ k ];

	int contador = 0;
	for ( CFCluster microcluster : microclusters) {
	    // Find closest kMeans cluster
	    double minDistance = Double.MAX_VALUE;
	    int closestCluster = 0;
	    for ( int i = 0; i < k; i++ ) {
			double distance = distance( kMeansResult.get(i).getCenter(), microcluster.getCenter() );
			if ( distance < minDistance ) {
			    closestCluster = i;
			    minDistance = distance;
			}
	    }

	    // Add to cluster
	    if ( res[closestCluster] == null ) {
	    	res[closestCluster] = (CFCluster)microcluster.copy();
	    } else {
	    	res[closestCluster].add(microcluster);
	    }
	    grupoAux.get(closestCluster).add((int)contador);
	    contador++;
	}

	// Clean up res
	int count = 0;
	for ( int i = 0; i < res.length; i++ ) {
	    if ( res[i] != null )
	    	++count;
	}

	CFCluster[] cleaned = new CFCluster[count];
	count = 0;
	for ( int i = 0; i < res.length; i++ ) {
	    if ( res[i] != null )
		cleaned[count++] = res[i];
	    grupo.add(grupoAux.get(i));
	}

	return new Clustering( cleaned );
    }
    
    
    public Clustering gaussianMeans4(Clustering clustering, int algEscolhaCentros,ArrayList<ArrayList<Integer>>grupoFinal) {
        ArrayList<CFCluster> microclusters = new ArrayList<CFCluster>();
        ArrayList<ArrayList<Integer>> vetGrupo[] ;
        ArrayList<ArrayList<Integer>>grupoAux;
        ArrayList<ArrayList<Integer>>grupo;
             
        for (int i = 0; i < clustering.size(); i++) {
            if (clustering.get(i) instanceof CFCluster) {
                microclusters.add((CFCluster)clustering.get(i));
            }
            else{
                System.out.println("Unsupported Cluster Type:"+clustering.get(i).getClass()
                        +". Cluster needs to extend moa.cluster.CFCluster");
            }
        }
        Cluster[] centers;
        Cluster[] melhoresCentros = null;
        ArrayList<CFCluster[]> ClusterFinal=new ArrayList<CFCluster[]>();
        ArrayList<Double> silhueta=new ArrayList<Double>();

        //feito por Elaine
        vetGrupo = (ArrayList<ArrayList<Integer>>[])new ArrayList[(int)Math.sqrt(microclusters.size())+1];
        for (int valorK=2; valorK<=(int)Math.sqrt(microclusters.size()); valorK++){
            grupo = new ArrayList<ArrayList<Integer>>();        	
        	grupoAux = new ArrayList<ArrayList<Integer>>();
            for (int i = 0; i < valorK; i++) {
               	grupoAux.add(new ArrayList<Integer>());
             }
        	
            centers = new Cluster[valorK];    
	        double CurCusto=0, MinCusto=0;
	        switch (algEscolhaCentros){
				//case 1: melhoresCentros = centrosMOA(microclusters, valorK, gtClustering); break;
				case 2: melhoresCentros = centrosAleatorios(microclusters, valorK); break; 
				case 3: melhoresCentros = centrosKMeansMais(valorK, microclusters.size(),microclusters.get(0).getCenter().length, microclusters); break;
				case 4:melhoresCentros = centrosClustream(microclusters, valorK); break;
	        }   	    
	        centers = melhoresCentros.clone();
	        int k = centers.length;
			if ( microclusters.size() < k ) {
			    return new Clustering( new Cluster[0]);
			}
			//feito por Elaine
			double Custo[] = new double[1];
		    Clustering kMeansResult = kMeans2( centers, microclusters, Custo );    
		    CurCusto = Custo[0];
		    MinCusto = CurCusto;
		    for (int i=1; i<5; i++){
		    	switch (algEscolhaCentros){
		    		//case 1: melhoresCentros = centrosMOA(microclusters, valorK, gtClustering); break;
		    		case 2: melhoresCentros = centrosAleatorios(microclusters, valorK); break; 
		    		case 3: melhoresCentros = centrosKMeansMais(valorK, microclusters.size(),microclusters.get(0).getCenter().length, microclusters); break;
		    		case 4:melhoresCentros = centrosClustream(microclusters, valorK); break;
		    	}    
		                      
		    	kMeansResult = kMeans2(melhoresCentros, microclusters, Custo );
		    	CurCusto = Custo[0];
		    	if(CurCusto < MinCusto) {
					MinCusto = CurCusto;
					//centers = melhoresCentros.clone();
					for (int m = 0; m<centers.length; m++)
						centers[m] = kMeansResult.get(m);
				}
		    }

			k = kMeansResult.size();
			CFCluster[] res = new CFCluster[ k ];
			int ResGrupos[] = new int[microclusters.size()];
			int contMicro = 0;
			
			int contador = 0;
			for ( CFCluster microcluster : microclusters) {
			    // Find closest kMeans cluster
			    double minDistance = Double.MAX_VALUE;
			    int closestCluster = 0;
			    for ( int i = 0; i < k; i++ ) {
					double distance = distance( centers[i].getCenter(), microcluster.getCenter() );
					if ( distance < minDistance ) {
					    closestCluster = i;
					    minDistance = distance;
					}
			    } 
			    ResGrupos[contMicro++] = closestCluster;
			    // Add to cluster
			    if ( res[closestCluster] == null ) {
				   res[closestCluster] = (CFCluster)microcluster.copy();
			    } else {
				   res[closestCluster].add(microcluster);
			    }
			    grupoAux.get(closestCluster).add((int)contador);
			    contador++;			    
			}

			// Clean up res
			int count = 0;
			for ( int i = 0; i < res.length; i++ ) {
			    if ( res[i] != null )
				++count;
			}
		
			CFCluster[] cleaned = new CFCluster[count];
			count = 0;
			for ( int i = 0; i < res.length; i++ ) {
			    if ( res[i] != null )
				    cleaned[count++] = res[i];
			    grupo.add(grupoAux.get(i));
			}	
			silhueta.add(new Double(silhueta(valorK,new Clustering(cleaned), ResGrupos, microclusters)));
	        ClusterFinal.add(cleaned);
	        vetGrupo[valorK]= new ArrayList<ArrayList<Integer>>();
	        vetGrupo[valorK] = grupo;
        }     
        double maior = silhueta.get(0);
        int posmaior = 0;
        for (int i=1; i<silhueta.size(); i++){
        	if (silhueta.get(i) > maior){
        		maior = silhueta.get(i);
        		posmaior = i;
        	}
        }
        melhorK = 2+posmaior;
        //verdadeiroK = gtClustering.size();
        //grupoFinal = new ArrayList<ArrayList<Integer>>();
        for (int m=0; m<vetGrupo[2+posmaior].size();m++)
        	grupoFinal.add(vetGrupo[2+posmaior].get(m));
		//grupoFinal= vetGrupo[2+posmaior];
		return new Clustering( ClusterFinal.get(posmaior) );
    }
    //*********************************************************************************
    public static Clustering OMRk(Clustering clustering, int algEscolhaCentros, ArrayList<ArrayList<Integer>>grupoFinal) {
    	double valorSilhueta = 0.0, maiorSilhueta = 0.0;
    	maiorSilhueta = Double.MAX_VALUE;
    	Clustering melhorParticao = new Clustering();
        ArrayList<CFCluster> microclusters = new ArrayList<CFCluster>();
        Cluster[] Centros = null;
        ArrayList<ArrayList<Integer>>grupo;
        
        for (int i = 0; i < clustering.size(); i++) {
            if (clustering.get(i) instanceof CFCluster) {
                microclusters.add((CFCluster)clustering.get(i));
            }
            else{
                System.out.println("Unsupported Cluster Type:"+clustering.get(i).getClass()
                        +". Cluster needs to extend moa.cluster.CFCluster");
            }
        }
        
    	for (int valorK=2; valorK<=(int)2*Math.sqrt(microclusters.size()); valorK++){
    		System.out.print("\n K:" + valorK + "  ");
    	 	for (int i=0; i<10; i++){
    	 		switch (algEscolhaCentros){    	 
	    	 		case 2: Centros = centrosAleatorios(microclusters, valorK); break; 
	    	 		case 3: Centros = centrosKMeansMais(valorK, microclusters.size(),microclusters.get(0).getCenter().length, microclusters); break;
	    	 		case 4: Centros = centrosClustream(microclusters, valorK); break;
	    	 		case 0: Centros = centrosSeq(microclusters, valorK);break;
    	 		}
    	 		double valor[] = new double[1];
    	 		Clustering kMeansResult = kMeansNaoVazio( Centros, microclusters, valor);
    	 		System.out.print(" v: " + valor[0]);
    	 		grupo = new ArrayList<ArrayList<Integer>>();
    	 		
    	 		//******
    	 		/*for (int j=0; j<kMeansResult.size(); j++)
    		    	grupo.add(new ArrayList<Integer>());
    		    
    		    for ( int p=0; p< microclusters.size(); p++) {
    				double minDistance = distance( microclusters.get(p).getCenter(), kMeansResult.get(0).getCenter() );
    				int closestCluster = 0;
    				for ( int j = 1; j < kMeansResult.size(); j++ ) {
    				    double distance = distance( microclusters.get(p).getCenter(), kMeansResult.get(j).getCenter() );
    				    if ( distance < minDistance ) {
    				    	closestCluster = j;
    				    	minDistance = distance;
    				    }
    				}
    				grupo.get(closestCluster).add(p);    					    	
    		    }*/
    	 		//******
    	 		valorSilhueta = silhueta(kMeansResult,microclusters, grupo);
    	 		//valorSilhueta = valor[0];
    	 		if ((valorK == 2) && (i==0))
    	 			maiorSilhueta = valorSilhueta; 
    	 		System.out.print(" s: " + valorSilhueta + " ");
    	 		if (valorSilhueta >= maiorSilhueta){
    	 		//if (valorSilhueta <= maiorSilhueta){
    	 			maiorSilhueta = valorSilhueta;
    	 			melhorK = valorK;
    	 			if (melhorParticao.size() > 0){
    	 				int tam = melhorParticao.size();
    	 				for (int j=0; j<tam;j++) 
    	 					melhorParticao.remove(0);
    	 			}
    	 			//if (grupoFinal.size() > 0){
    	 				//int tam = grupoFinal.size();
    	 				//for (int j=0; j<tam; j++)
    	 					//grupoFinal.remove(0);
    	 			//}
    	 			for (int j=0; j<kMeansResult.size();j++){
    	 				melhorParticao.add(kMeansResult.get(j));
    	 				//grupoFinal.add(grupo.get(j));
    	 			}
    	 		}    	 		
    	 	}
    	}
    	System.out.println("\n Melhor K: " + melhorK);
    	ArrayList<ArrayList<Integer>>grupoAux = new ArrayList<ArrayList<Integer>>();
    	for (int i = 0; i < melhorK; i++) {
           	grupoAux.add(new ArrayList<Integer>());
         }
    	
    	CFCluster[] res = new CFCluster[ melhorK ];
    	int contador=0;
    	for ( CFCluster microcluster : microclusters) {
		    // Find closest kMeans cluster
		    double minDistance = Double.MAX_VALUE;
		    int closestCluster = 0;

		    for ( int i = 0; i < melhorK; i++ ) {
				double distance = distance( melhorParticao.get(i).getCenter(), microcluster.getCenter() );

				if ( distance < minDistance ) {
				    closestCluster = i;
				    minDistance = distance;
				}
		    }
		    //ResGrupos[contMicro++] = closestCluster;
		    // Add to cluster
		    if ( res[closestCluster] == null ) {
		    	res[closestCluster] = (CFCluster)microcluster.copy();
		    } else {
		    	res[closestCluster].add(microcluster);
		    }
		    grupoAux.get(closestCluster).add((int)contador);
		    contador++;	
		}

		// Clean up res
		int count = 0;
		for ( int i = 0; i < res.length; i++ ) {
		    if ( res[i] != null )
		    	++count;
		}
	
		CFCluster[] cleaned = new CFCluster[count];
		count = 0;
		for ( int i = 0; i < res.length; i++ ) {
		    if ( res[i] != null ){
		    	cleaned[count++] = res[i];
		    	grupoFinal.add(grupoAux.get(i));
		    }
		}	
     
	return new Clustering(cleaned);
	
    }

  //***************************

    
    public static Cluster[] centrosMOA(ArrayList<CFCluster> clustering, int k, Clustering clusterTrue){
    	
    	Cluster[] centers = new Cluster[k];
    	for (int i=0; i<centers.length; i++)
    		centers[i] = clusterTrue.get(i);
    	return centers;
    }
    private static Cluster[] centrosClustream(ArrayList<CFCluster> microclusters, int tam){
        //Modo1: centros com probabilidade proporcional ao tamanho dos grupos
        Cluster[] centers = new Cluster[tam];
        //ordena os microclusters em ordem decrescente de tamanho
         Collections.sort (microclusters, new Comparator() {  
             public int compare(Object o1, Object o2) {  
            	 CFCluster p1 = (CFCluster) o1;  
            	 CFCluster p2 = (CFCluster) o2;  
                 return p1.getWeight() > p2.getWeight() ? -1 : (p1.getWeight() < p2.getWeight() ? +1 : 0);  
             }  
         });  
         int tamanhoTotal=0;
         //calculo do tamanho total de elementos nos microclusters
         for (int i=0; i<microclusters.size(); i++){
         	tamanhoTotal+= microclusters.get(i).getWeight();
         }
         MTRandom m = new MTRandom(System.currentTimeMillis());
         //MTRandom m = new MTRandom(1);
         int nroaleatorio=0, nromicro=0;
         int nrocentros = centers.length;
         ArrayList<Integer>NroSorteados = new ArrayList<Integer>();
         for (int i = 0; i < nrocentros; i++) {
         	nroaleatorio = m.nextInt(tamanhoTotal);
         	//busca em qual microclusters esta o nro aleatorio
         	int soma = 0;
         	for (int j=0; j<microclusters.size(); j++){
         		if ((nroaleatorio >= soma) && (nroaleatorio < microclusters.get(j).getWeight()+soma)){
         				if (!(NroSorteados.contains(j))){
         					nromicro = j;
         					NroSorteados.add(nromicro);
         					break;
         				}
         				else{
         					i--;
         					break;
         				}
         		}
         		soma+= (int) microclusters.get(j).getWeight();
         	}        	
         	centers[i] = microclusters.get(nromicro);
         }
         return centers;
    }

    public static Cluster[]  centrosKMeansMais(int k, int n, int d, ArrayList<CFCluster> points){

		//array to store the choosen centres
    	Cluster[]  centres = new Cluster[k]; 
    	MTRandom clustererRandom;
    	clustererRandom = new MTRandom(System.currentTimeMillis());
    	int centreIndex[] = new int[points.size()];
    	double curCost []= new double[points.size()];

		//choose the first centre (each point has the same probability of being choosen)
		int i = 0;
		
		int next = 0;
		int j = 0;
		//do{ //only choose from the n-i points not already choosen
			next = clustererRandom.nextInt(n-1); 
			
			//check if the choosen point is not a dummy
		//} while( points[next].weight < 1);
		
		//set j to next unchoosen point
		j = next;
		//copy the choosen point to the array
		centres[i] = points.get(j);
			
		//set the current centre for all points to the choosen centre
		for(i = 0; i < n; i++){
			centreIndex[i] = 0;
			curCost[i] = distance(points.get(i).getCenter(), centres[0].getCenter()); 		
		}
		//choose centre 1 to k-1 with the kMeans++ distribution
		for(i = 1; i < k; i++){

			double cost = 0.0;
			for(j = 0; j < n; j++){
				cost += curCost[j];
			}
			
			double random = 0;
			double sum = 0.0;
			int pos = -1;
			
			//do{
					random = clustererRandom.nextDouble();//genrand_real3();
					sum = 0.0;
					pos = -1;

				for(j = 0; j < n; j++){
					sum = sum + curCost[j];
					if(random <= sum/cost){
						pos = j;
						break;
					}	
				}	
			//} while (points[pos].weight < 1);
				
			//copy the choosen centre
			centres[i] = points.get(pos);
			//check which points are closest to the new centre
			for(j = 0; j < n; j++){
				double newCost = distance(points.get(j).getCenter(),centres[i].getCenter());
				if(curCost[j] > newCost){
					curCost[j] = newCost;
					centreIndex[j] = i;
				}
			}
			
		}
		
		return centres;
	}

    public static Cluster[] centrosAleatorios(ArrayList<CFCluster> clustering, int k){
    	int nroaleatorio;
        List<Integer> numeros = new ArrayList<Integer>();  
        for (int cont = 0; cont < clustering.size(); cont++) { 
            numeros.add(cont);  
        }  
        Collections.shuffle(numeros); 
        Cluster[] centers = new Cluster[k];
        for (int i = 0; i < centers.length; i++) {
        	nroaleatorio = numeros.get(i).intValue();
        	System.out.print(nroaleatorio + " ");
            centers[i] = clustering.get(nroaleatorio);
        }
        return centers;
        
        
	  /*  Random obj = new Random();
	    obj.setSeed(10);
    	Cluster[] centers = new Cluster[k];
    	HashSet<Integer> valores = new HashSet<Integer>();  
	    while (valores.size() < centers.length) {    
	         int x = (int) (obj.nextInt(clustering.size()));
	         valores.add(new Integer(x));    
	    }  
	    int cont = 0;
	    for (Integer i : valores) {
    	   centers[cont] = clustering.get(i);
    	   cont++;
	    }
       return centers;*/
    }

    public static Cluster[] centrosSeq(ArrayList<CFCluster> clustering, int k){
        Cluster[] centers = new Cluster[k];
       
        for (int i = 0; i < centers.length; i++) {
        		centers[i] = clustering.get(i);
        }                	
        return centers;
    }
    
    public static double silhueta(Clustering clusters, ArrayList<CFCluster> microclusters,ArrayList<ArrayList<Integer>>grupo ){
	    int numFCluster = clusters.size();
	    double silhueta=0;
	    double silhuetaPonto = 0;    
	    int ResGrupos[] = new int[microclusters.size()];
	    
	    for (int i=0; i<numFCluster; i++)
	    	grupo.add(new ArrayList<Integer>());
	    
	    for ( int i=0; i< microclusters.size(); i++) {
			double minDistance = distance( microclusters.get(i).getCenter(), clusters.get(0).getCenter() );
			int closestCluster = 0;
			for ( int j = 1; j < numFCluster; j++ ) {
			    double distance = distance( microclusters.get(i).getCenter(), clusters.get(j).getCenter() );
			    if ( distance < minDistance ) {
			    	closestCluster = j;
			    	minDistance = distance;
			    }
			}
			grupo.get(closestCluster).add(i);
	    	ResGrupos[i] = closestCluster;
				    	
	    }
	    	
	    for (int p=0; p<ResGrupos.length;p++){
	    	Integer ownClusters=ResGrupos[p];
	    	    		
	       	double[] distanceByClusters = new double[numFCluster-1];
	       	//laco para procurar a distancia de cada elemento a todos os grupos
	       	int pos=0;
	       	for (int fc = 0; fc < numFCluster; fc++) {
	       		if (fc!=ownClusters){
	       			double distance = distance(microclusters.get(p).getCenter(), clusters.get(fc).getCenter());
	       			distanceByClusters[pos] = distance;
	       			pos++;
	       		}
	       	}
	       	Arrays.sort(distanceByClusters);
	       	double distPontoSeuCluster = distance(microclusters.get(p).getCenter(),clusters.get(ownClusters).getCenter());
	       	double distPontoClusterMaisProx = distanceByClusters[0];
	
	       	silhuetaPonto += (distPontoClusterMaisProx - distPontoSeuCluster)/Math.max(distPontoClusterMaisProx, distPontoSeuCluster);
	    }
	    silhueta = silhuetaPonto / microclusters.size(); 
	    return silhueta;
    }  	

    public static double silhueta(int k, Clustering clusters, int ResGrupos[],ArrayList<CFCluster> microclusters ){
	    int numFCluster = k;
	    double silhueta=0;
	    double silhuetaPonto = 0;
	
	    for (int p=0; p<ResGrupos.length;p++){
	    	Integer ownClusters=ResGrupos[p];
	    	    		
	       	double[] distanceByClusters = new double[numFCluster-1];
	       	//laco para procurar a distancia de cada elemento a todos os grupos
	       	int pos=0;
	       	for (int fc = 0; fc < numFCluster; fc++) {
	       		if (fc!=ownClusters){
	       			double distance = distance(microclusters.get(p).getCenter(), clusters.get(fc).getCenter());
	       			distanceByClusters[pos] = distance;
	       			pos++;
	       		}
	       	}
	       	Arrays.sort(distanceByClusters);
	       	double distPontoSeuCluster = distance(microclusters.get(p).getCenter(),clusters.get(ownClusters).getCenter());
	       	double distPontoClusterMaisProx = distanceByClusters[0];
	
	       	silhuetaPonto += (distPontoClusterMaisProx - distPontoSeuCluster)/Math.max(distPontoClusterMaisProx, distPontoSeuCluster);
	    }
	    silhueta = silhuetaPonto / microclusters.size(); 
	    return silhueta;
    }  	

    
}
