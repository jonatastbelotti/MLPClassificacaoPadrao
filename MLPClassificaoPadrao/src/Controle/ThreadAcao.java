package Controle;

import Modelo.MLP;
import Recursos.Arquivo;
import javax.swing.JOptionPane;

/**
 *
 * @author Jônatas Trabuco Belotti [jonatas.t.belotti@hotmail.com]
 */
public class ThreadAcao extends Thread {

  public static final int TREINAR_NORMAL = 0;
  public static final int TREINAR_MOMENTUM = 1;

  private MLP redeMLP = null;
  private Arquivo arquivoTreinamento;
  private int tipoTreinamento;

  public ThreadAcao(MLP redeMLP) {
    this.redeMLP = redeMLP;
  }

  public void setArquivoTreinamento(Arquivo arquivoTreinamento) {
    this.arquivoTreinamento = arquivoTreinamento;
  }

  public void setTipoTreinamento(int tipoTreinamento) {
    this.tipoTreinamento = tipoTreinamento;
  }

  @Override
  public void run() {
    if (redeMLP != null && arquivoTreinamento != null) {
      if (tipoTreinamento == TREINAR_MOMENTUM) {
        imprimirMensagem(redeMLP.treinarMomentum(arquivoTreinamento));
      } else {
        imprimirMensagem(redeMLP.treinar(arquivoTreinamento));
      }

      stop();
    }
  }

  private void imprimirMensagem(boolean val) {
    if (val) {
      JOptionPane.showMessageDialog(null, "Rede MLP treinada com sucesso!", "Sucesso", JOptionPane.INFORMATION_MESSAGE);
      Comunicador.setEnabledBotaoTestar(true);
      Comunicador.setEnabledBotaoSalvar(true);
    } else {
      JOptionPane.showMessageDialog(null, "Houve um erro no treinamento da rede!", "Erro", JOptionPane.ERROR_MESSAGE);
      Comunicador.setEnabledBotaoTestar(false);
      Comunicador.setEnabledBotaoSalvar(false);
    }
  }

}
