package Modelo;

import Controle.Comunicador;
import Recursos.Arquivo;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

/**
 *
 * @author Jônatas Trabuco Belotti [jonatas.t.belotti@hotmail.com]
 */
public class MLP {

  public static final int NUM_ENTRADAS = 4;
  private final int NUM_NEU_CAMADA_ESCONDIDA = 15;
  public static final int NUM_NEU_CAMADA_SAIDA = 3;
  private final double TAXA_APRENDIZAGEM = 0.1;
  private final double PRECISAO = 0.000001;
  private final double FATOR_MOMENTUM = 0.9;
  private final double BETA = 1.0;

  private int numEpocas;
  private double fatorMomentum;
  private double[] entradas;
  private double[][] pesosCamadaEscondidaInicial;
  private double[][] pesosCamadaEscondida;
  private double[][] pesosCamadaEscondidaProximo;
  private double[][] pesosCamadaEscondidaAnterior;
  private double[][] pesosCamadaSaidaInicial;
  private double[][] pesosCamadaSaida;
  private double[][] pesosCamadaSaidaProximo;
  private double[][] pesosCamadaSaidaAnterior;
  private double[] potencialCamadaEscondida;
  private double[] saidaCamadaEscondida;
  private double[] potencialCamadaSaida;
  private double[] saidaCamadaSaida;
  private double[] saidaEsperada;
  private double[] gradienteCamadaSaida;
  private double[] gradienteCamadaEscondida;

  public MLP() {
    Random random;

    entradas = new double[NUM_ENTRADAS + 1];
    pesosCamadaEscondidaInicial = new double[NUM_NEU_CAMADA_ESCONDIDA][NUM_ENTRADAS + 1];
    pesosCamadaEscondida = new double[NUM_NEU_CAMADA_ESCONDIDA][NUM_ENTRADAS + 1];
    pesosCamadaEscondidaAnterior = new double[NUM_NEU_CAMADA_ESCONDIDA][NUM_ENTRADAS + 1];
    pesosCamadaEscondidaProximo = new double[NUM_NEU_CAMADA_ESCONDIDA][NUM_ENTRADAS + 1];
    pesosCamadaSaidaInicial = new double[NUM_NEU_CAMADA_SAIDA][NUM_NEU_CAMADA_ESCONDIDA + 1];
    pesosCamadaSaida = new double[NUM_NEU_CAMADA_SAIDA][NUM_NEU_CAMADA_ESCONDIDA + 1];
    pesosCamadaSaidaAnterior = new double[NUM_NEU_CAMADA_SAIDA][NUM_NEU_CAMADA_ESCONDIDA + 1];
    pesosCamadaSaidaProximo = new double[NUM_NEU_CAMADA_SAIDA][NUM_NEU_CAMADA_ESCONDIDA + 1];
    potencialCamadaEscondida = new double[NUM_NEU_CAMADA_ESCONDIDA + 1];
    saidaCamadaEscondida = new double[NUM_NEU_CAMADA_ESCONDIDA + 1];
    potencialCamadaSaida = new double[NUM_NEU_CAMADA_SAIDA];
    saidaCamadaSaida = new double[NUM_NEU_CAMADA_SAIDA];
    saidaEsperada = new double[NUM_NEU_CAMADA_SAIDA];
    gradienteCamadaSaida = new double[NUM_NEU_CAMADA_SAIDA];
    gradienteCamadaEscondida = new double[NUM_NEU_CAMADA_ESCONDIDA];

    //Iniciando pesos sinapticos
    random = new Random();

    for (int i = 0; i < NUM_NEU_CAMADA_ESCONDIDA; i++) {
      for (int j = 0; j < NUM_ENTRADAS + 1; j++) {
        pesosCamadaEscondidaInicial[i][j] = random.nextDouble();
      }
    }

    for (int i = 0; i < NUM_NEU_CAMADA_SAIDA; i++) {
      for (int j = 0; j < NUM_NEU_CAMADA_ESCONDIDA + 1; j++) {
        pesosCamadaSaidaInicial[i][j] = random.nextDouble();
      }
    }
  }

  public boolean treinar(Arquivo arquivoTreinamento) {
    Comunicador.limparLog();
    return treinar(arquivoTreinamento, false);
  }
  
  public boolean treinar(Arquivo arquivoTreinamento, boolean momentum) {
    FileReader arq;
    BufferedReader lerArq;
    String linha;
    double erroAtual;
    double erroAnterior;
    long tempInicial;

    copiarMatriz(pesosCamadaEscondidaInicial, pesosCamadaEscondida);
    copiarMatriz(pesosCamadaSaidaInicial, pesosCamadaSaida);
    copiarMatriz(pesosCamadaEscondidaInicial, pesosCamadaEscondidaAnterior);
    copiarMatriz(pesosCamadaSaidaInicial, pesosCamadaSaidaAnterior);

    tempInicial = System.currentTimeMillis();
    numEpocas = 0;
    erroAtual = erroQuadraticoMedio(arquivoTreinamento);
    
    if (!momentum) {
      fatorMomentum = 0D;
      Comunicador.addLog("Início treinamento da MLP");
    } else {
      fatorMomentum = FATOR_MOMENTUM;
      Comunicador.addLog("--------------------------------------");
      Comunicador.addLog("Início treinamento com MOMENTUM da MLP");
    }

    Comunicador.addLog(String.format("Erro inicial: %.6f", erroAtual).replace(".", ","));
    imprimirPesos();
    Comunicador.addLog("Época Eqm");

    try {
      do {
        this.numEpocas++;
        erroAnterior = erroAtual;
        arq = new FileReader(arquivoTreinamento.getCaminhoCompleto());
        lerArq = new BufferedReader(arq);

        linha = lerArq.readLine();
        if (linha.contains("x1")) {
          linha = lerArq.readLine();
        }

        while (linha != null) {
          separarEntradas(linha);

          calcularSaidas();

          ajustarPesos();

          linha = lerArq.readLine();
        }

        arq.close();
        erroAtual = erroQuadraticoMedio(arquivoTreinamento);
        Comunicador.addLog(String.format("%d   %.6f", numEpocas, erroAtual).replace(".", ","));
      } while (Math.abs(erroAtual - erroAnterior) > PRECISAO && numEpocas < 10000);

      Comunicador.addLog(String.format("Fim do treinamento. (%.2fs)", (double) (System.currentTimeMillis() - tempInicial) / 1000D));
      imprimirPesos();
    } catch (FileNotFoundException ex) {
      return false;
    } catch (IOException ex) {
      return false;
    }

    return true;
  }

  public boolean treinarMomentum(Arquivo arquivoTreinamento) {
    if (!treinar(arquivoTreinamento, false)) {
      return false;
    }
    
    if (!treinar(arquivoTreinamento, true)) {
      return false;
    }
    
    return true;
  }

  public void testar(Arquivo arquivoTreinamento) {
    FileReader arq;
    BufferedReader lerArq;
    String linha;
    String esperada;
    String resposta;

    int erros = 0;

    Comunicador.iniciarLog("Início teste da MLP");
    Comunicador.addLog("Resposta - Saída rede          Erro");

    try {
      arq = new FileReader(arquivoTreinamento.getCaminhoCompleto());
      lerArq = new BufferedReader(arq);

      linha = lerArq.readLine();
      if (linha.contains("x1")) {
        linha = lerArq.readLine();
      }

      while (linha != null) {
        separarEntradas(linha);

        calcularSaidas();

        esperada = "";
        resposta = "";
        for (int i = 0; i < NUM_NEU_CAMADA_SAIDA; i++) {
          esperada += String.format("%.0f ", saidaEsperada[i]);
          resposta += String.format("%d ", posProcessamento(saidaCamadaSaida[i]));

          if (saidaEsperada[i] != posProcessamento(saidaCamadaSaida[i])) {
            erros++;
          }
        }

        Comunicador.addLog(esperada + "- " + resposta);

        linha = lerArq.readLine();
      }

      arq.close();
      Comunicador.addLog(String.format("Erros: %d", erros));
    } catch (FileNotFoundException ex) {
    } catch (IOException ex) {
    }
  }

  private double erroQuadraticoMedio(Arquivo arquivo) {
    FileReader arq;
    BufferedReader lerArq;
    String linha;
    int numAmostras;
    double erro;
    double valorParcial;

    erro = 0D;
    numAmostras = 0;

    try {
      arq = new FileReader(arquivo.getCaminhoCompleto());
      lerArq = new BufferedReader(arq);

      linha = lerArq.readLine();
      if (linha.contains("x1")) {
        linha = lerArq.readLine();
      }

      while (linha != null) {
        numAmostras++;
        separarEntradas(linha);

        calcularSaidas();

        //Calculando erro
        valorParcial = 0D;
        for (int i = 0; i < saidaCamadaSaida.length; i++) {
          valorParcial = valorParcial + Math.pow((double) (saidaEsperada[i] - saidaCamadaSaida[i]), 2D);
        }
        erro = erro + (valorParcial / 2D);

        linha = lerArq.readLine();
      }

      arq.close();
      erro = erro / (double) numAmostras;

    } catch (FileNotFoundException ex) {
    } catch (IOException ex) {
    }

    return erro;
  }

  private void separarEntradas(String linha) {
    String[] vetor;
    int i;

    vetor = linha.split("\\s+");
    i = 0;

    if (vetor[0].equals("")) {
      i = 1;
    }

    entradas[0] = -1.0;
    for (int j = 1; j <= NUM_ENTRADAS; j++) {
      entradas[j] = Double.parseDouble(vetor[i++].replace(",", "."));
    }
    for (int j = 0; j < NUM_NEU_CAMADA_SAIDA; j++) {
      saidaEsperada[j] = Double.parseDouble(vetor[i++].replace(",", "."));
    }
  }

  private void calcularSaidas() {
    double valorParcial;

    //Calculando saidas da camada escondida
    saidaCamadaEscondida[0] = -1D;
    potencialCamadaEscondida[0] = -1D;

    for (int i = 1; i < saidaCamadaEscondida.length; i++) {
      valorParcial = 0D;

      for (int j = 0; j < entradas.length; j++) {
        valorParcial += entradas[j] * pesosCamadaEscondida[i - 1][j];
      }

      potencialCamadaEscondida[i] = valorParcial;
      saidaCamadaEscondida[i] = funcaoLogistica(valorParcial);
    }

    //Calculando saida da camada de saída
    for (int i = 0; i < saidaCamadaSaida.length; i++) {
      valorParcial = 0D;

      for (int j = 0; j < saidaCamadaEscondida.length; j++) {
        valorParcial += saidaCamadaEscondida[j] * pesosCamadaSaida[i][j];
      }

      potencialCamadaSaida[i] = valorParcial;
      saidaCamadaSaida[i] = funcaoLogistica(valorParcial);
    }
  }

  private void ajustarPesos() {
    //Ajustando pesos sinapticos da camada de saida
    for (int i = 0; i < gradienteCamadaSaida.length; i++) {
      gradienteCamadaSaida[i] = ((double) saidaEsperada[i] - (double) saidaCamadaSaida[i]) * funcaoLogisticaDerivada(saidaCamadaSaida[i]);

      for (int j = 0; j < NUM_NEU_CAMADA_ESCONDIDA + 1; j++) {
        pesosCamadaSaidaProximo[i][j] = pesosCamadaSaida[i][j] + (fatorMomentum * (pesosCamadaSaida[i][j] - pesosCamadaSaidaAnterior[i][j])) + (TAXA_APRENDIZAGEM * gradienteCamadaSaida[i] * saidaCamadaEscondida[j]);
      }
    }

    //Ajustando pesos sinapticos da camada escondida
    for (int i = 0; i < gradienteCamadaEscondida.length; i++) {
      gradienteCamadaEscondida[i] = 0D;
      for (int j = 0; j < NUM_NEU_CAMADA_SAIDA; j++) {
        gradienteCamadaEscondida[i] += gradienteCamadaSaida[j] * pesosCamadaSaidaProximo[j][i + 1] * funcaoLogisticaDerivada(potencialCamadaEscondida[i + 1]);
      }

      for (int j = 0; j < NUM_ENTRADAS + 1; j++) {
        pesosCamadaEscondidaProximo[i][j] = pesosCamadaEscondida[i][j] + (fatorMomentum * (pesosCamadaEscondida[i][j] - pesosCamadaEscondidaAnterior[i][j])) + (TAXA_APRENDIZAGEM * gradienteCamadaEscondida[i] * entradas[j]);
      }
    }
    
    //Copiando pesos
    copiarMatriz(pesosCamadaEscondida, pesosCamadaEscondidaAnterior);
    copiarMatriz(pesosCamadaSaida, pesosCamadaSaidaAnterior);
    copiarMatriz(pesosCamadaEscondidaProximo, pesosCamadaEscondida);
    copiarMatriz(pesosCamadaSaidaProximo, pesosCamadaSaida);
  }

  private void copiarMatriz(double[][] origem, double[][] destino) {
    for (int i = 0; i < origem.length; i++) {
      for (int j = 0; j < origem[i].length; j++) {
        if (destino.length > i) {
          if (destino[i].length > j) {
            destino[i][j] = origem[i][j];
          }
        }
      }
    }
  }

  private double funcaoLogistica(double valor) {
    return 1D / (1D + Math.pow(Math.E, -1D * BETA * valor));
  }

  private double funcaoLogisticaDerivada(double valor) {
    return (BETA * Math.pow(Math.E, -1D * BETA * valor)) / Math.pow((Math.pow(Math.E, -1D * BETA * valor) + 1D), 2D);
  }

  private int posProcessamento(double valor) {
    int resposta = 0;

    if (valor >= 0.5) {
      resposta = 1;
    }

    return resposta;
  }

  private void imprimirPesos() {
    String log;

    Comunicador.addLog("Pesos camada escondida:");

    for (int i = 0; i < NUM_NEU_CAMADA_ESCONDIDA; i++) {
      log = "N" + (i + 1) + " =";

      for (int j = 0; j < NUM_ENTRADAS + 1; j++) {
        log += String.format(" %f", pesosCamadaEscondida[i][j]);
      }

      Comunicador.addLog(log);
    }

    Comunicador.addLog("Pesos camada de saída:");
    for (int i = 0; i < NUM_NEU_CAMADA_SAIDA; i++) {
      log = "N" + (i + 1) + " =";

      for (int j = 0; j < NUM_NEU_CAMADA_ESCONDIDA + 1; j++) {
        log += String.format(" %f", pesosCamadaSaida[i][j]);
      }

      Comunicador.addLog(log);
    }
  }

}
