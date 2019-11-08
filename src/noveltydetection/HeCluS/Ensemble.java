/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package noveltydetection.HeCluS;

/**
 *
 * @author kemilly
 */
public class Ensemble {
    
    private String[] nome = new String[2]; //nome do algoritmo de agrupamento
    private int qtd; // quantidade de vezes para execução
    
    public Ensemble(String[] nome, int qtd){
        this.nome = nome;
        this.qtd = qtd;
    }
    
    public Ensemble(){
        
    }

    public String getNome(int pos) {
        return nome[pos];
    }
    
    public String[] getNomes() {
        return nome;
    }

    public void setNome(int pos, String nome) {
        this.nome[pos] = nome;
    }

    public int getQtd() {
        return qtd;
    }

    public void setQtd(int qtd) {
        this.qtd = qtd;
    }
    
    
    
}
