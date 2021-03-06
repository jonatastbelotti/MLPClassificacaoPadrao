package Visao;

import Controle.Comunicador;
import Controle.ThreadAcao;
import Modelo.MLP;
import Recursos.Arquivo;
import java.awt.FileDialog;
import javax.swing.JOptionPane;

/**
 *
 * @author Jônatas Trabuco Belotti [jonatas.t.belotti@hotmail.com]
 */
public class Main extends javax.swing.JFrame {

  private FileDialog janelaAbrir;
  private FileDialog janelaSalvar;
  private Arquivo arquivoTreinamento;
  private Arquivo arquivoTeste;
  private MLP redeMLP;

  /**
   * Creates new form Main
   */
  public Main() {
    initComponents();
    this.setLocationRelativeTo(null);
    setExtendedState(MAXIMIZED_BOTH);
  }

  /**
   * This method is called from within the constructor to initialize the form.
   * WARNING: Do NOT modify this code. The content of this method is always
   * regenerated by the Form Editor.
   */
  @SuppressWarnings("unchecked")
  // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
  private void initComponents() {

    jPnAbrirArquivo = new javax.swing.JPanel();
    jLabel2 = new javax.swing.JLabel();
    jTxtArquivoDadosTreinamento = new javax.swing.JTextField();
    jBtnAbrirDadosTreinamento = new javax.swing.JButton();
    jLabel1 = new javax.swing.JLabel();
    jBtnTreinar = new javax.swing.JButton();
    jScrollPane1 = new javax.swing.JScrollPane();
    jTxtLogTreinamento = new javax.swing.JTextArea();
    jLabel3 = new javax.swing.JLabel();
    jBtnSalvarTreinamento = new javax.swing.JButton();
    jBtnTreinarMomentum = new javax.swing.JButton();
    jBtnIniciarPesos = new javax.swing.JButton();
    jPnTreinamento = new javax.swing.JPanel();
    jLabel6 = new javax.swing.JLabel();
    jLabel4 = new javax.swing.JLabel();
    jTxtArquivoDadosTeste = new javax.swing.JTextField();
    jBtnAbrirDadosTeste = new javax.swing.JButton();
    jBtnTestar = new javax.swing.JButton();
    jScrollPane3 = new javax.swing.JScrollPane();
    jTxtLogTeste = new javax.swing.JTextArea();
    jBtnSalvarTeste = new javax.swing.JButton();

    setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
    setTitle("MLP para classificação de padrões - Jônatas Trabuco Belotti");
    setMinimumSize(new java.awt.Dimension(730, 700));

    jPnAbrirArquivo.setBorder(javax.swing.BorderFactory.createEtchedBorder());

    jLabel2.setText("Selecione o arquivo com os dados de treinamento:");

    jTxtArquivoDadosTreinamento.setEditable(false);
    jTxtArquivoDadosTreinamento.setFocusable(false);

    jBtnAbrirDadosTreinamento.setText("Abrir");
    jBtnAbrirDadosTreinamento.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(java.awt.event.ActionEvent evt) {
        jBtnAbrirDadosTreinamentoActionPerformed(evt);
      }
    });

    jLabel1.setFont(new java.awt.Font("Dialog", 1, 18)); // NOI18N
    jLabel1.setText("Treinamento");

    jBtnTreinar.setText("Treinar");
    jBtnTreinar.setEnabled(false);
    jBtnTreinar.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(java.awt.event.ActionEvent evt) {
        jBtnTreinarActionPerformed(evt);
      }
    });

    jTxtLogTreinamento.setEditable(false);
    jTxtLogTreinamento.setColumns(20);
    jTxtLogTreinamento.setRows(5);
    jScrollPane1.setViewportView(jTxtLogTreinamento);

    jLabel3.setText("* Limitado em 10.000 épocas");

    jBtnSalvarTreinamento.setText("Salvar");
    jBtnSalvarTreinamento.setEnabled(false);
    jBtnSalvarTreinamento.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(java.awt.event.ActionEvent evt) {
        jBtnSalvarTreinamentoActionPerformed(evt);
      }
    });

    jBtnTreinarMomentum.setText("Treinar com momentum");
    jBtnTreinarMomentum.setEnabled(false);
    jBtnTreinarMomentum.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(java.awt.event.ActionEvent evt) {
        jBtnTreinarMomentumActionPerformed(evt);
      }
    });

    jBtnIniciarPesos.setText("Iniciar pesos");
    jBtnIniciarPesos.setEnabled(false);
    jBtnIniciarPesos.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(java.awt.event.ActionEvent evt) {
        jBtnIniciarPesosActionPerformed(evt);
      }
    });

    javax.swing.GroupLayout jPnAbrirArquivoLayout = new javax.swing.GroupLayout(jPnAbrirArquivo);
    jPnAbrirArquivo.setLayout(jPnAbrirArquivoLayout);
    jPnAbrirArquivoLayout.setHorizontalGroup(
      jPnAbrirArquivoLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
      .addGroup(jPnAbrirArquivoLayout.createSequentialGroup()
        .addContainerGap()
        .addGroup(jPnAbrirArquivoLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
          .addGroup(jPnAbrirArquivoLayout.createSequentialGroup()
            .addGroup(jPnAbrirArquivoLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
              .addGroup(jPnAbrirArquivoLayout.createSequentialGroup()
                .addGroup(jPnAbrirArquivoLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                  .addComponent(jLabel1)
                  .addComponent(jLabel2))
                .addGap(0, 0, Short.MAX_VALUE))
              .addComponent(jTxtArquivoDadosTreinamento))
            .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
            .addComponent(jBtnAbrirDadosTreinamento, javax.swing.GroupLayout.PREFERRED_SIZE, 70, javax.swing.GroupLayout.PREFERRED_SIZE))
          .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPnAbrirArquivoLayout.createSequentialGroup()
            .addComponent(jLabel3)
            .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addComponent(jBtnSalvarTreinamento)
            .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
            .addComponent(jBtnIniciarPesos, javax.swing.GroupLayout.PREFERRED_SIZE, 125, javax.swing.GroupLayout.PREFERRED_SIZE)
            .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
            .addComponent(jBtnTreinar, javax.swing.GroupLayout.PREFERRED_SIZE, 150, javax.swing.GroupLayout.PREFERRED_SIZE)
            .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
            .addComponent(jBtnTreinarMomentum))
          .addComponent(jScrollPane1))
        .addContainerGap())
    );
    jPnAbrirArquivoLayout.setVerticalGroup(
      jPnAbrirArquivoLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
      .addGroup(jPnAbrirArquivoLayout.createSequentialGroup()
        .addContainerGap()
        .addComponent(jLabel1)
        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
        .addComponent(jLabel2)
        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
        .addGroup(jPnAbrirArquivoLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
          .addComponent(jTxtArquivoDadosTreinamento, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
          .addComponent(jBtnAbrirDadosTreinamento))
        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
        .addGroup(jPnAbrirArquivoLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
          .addGroup(jPnAbrirArquivoLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
            .addComponent(jBtnTreinar)
            .addComponent(jBtnSalvarTreinamento)
            .addComponent(jBtnTreinarMomentum)
            .addComponent(jBtnIniciarPesos))
          .addComponent(jLabel3))
        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
        .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 244, Short.MAX_VALUE)
        .addContainerGap())
    );

    jPnTreinamento.setBorder(javax.swing.BorderFactory.createEtchedBorder());

    jLabel6.setFont(new java.awt.Font("Dialog", 1, 18)); // NOI18N
    jLabel6.setText("Teste");

    jLabel4.setText("Selecione o arquivo com os dados de teste:");

    jTxtArquivoDadosTeste.setEditable(false);
    jTxtArquivoDadosTeste.setFocusable(false);

    jBtnAbrirDadosTeste.setText("Abrir");
    jBtnAbrirDadosTeste.setEnabled(false);
    jBtnAbrirDadosTeste.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(java.awt.event.ActionEvent evt) {
        jBtnAbrirDadosTesteActionPerformed(evt);
      }
    });

    jBtnTestar.setText("Testar");
    jBtnTestar.setEnabled(false);
    jBtnTestar.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(java.awt.event.ActionEvent evt) {
        jBtnTestarActionPerformed(evt);
      }
    });

    jTxtLogTeste.setEditable(false);
    jTxtLogTeste.setColumns(20);
    jTxtLogTeste.setRows(5);
    jScrollPane3.setViewportView(jTxtLogTeste);

    jBtnSalvarTeste.setText("Salvar");
    jBtnSalvarTeste.setEnabled(false);
    jBtnSalvarTeste.addActionListener(new java.awt.event.ActionListener() {
      public void actionPerformed(java.awt.event.ActionEvent evt) {
        jBtnSalvarTesteActionPerformed(evt);
      }
    });

    javax.swing.GroupLayout jPnTreinamentoLayout = new javax.swing.GroupLayout(jPnTreinamento);
    jPnTreinamento.setLayout(jPnTreinamentoLayout);
    jPnTreinamentoLayout.setHorizontalGroup(
      jPnTreinamentoLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
      .addGroup(jPnTreinamentoLayout.createSequentialGroup()
        .addContainerGap()
        .addGroup(jPnTreinamentoLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
          .addGroup(jPnTreinamentoLayout.createSequentialGroup()
            .addGroup(jPnTreinamentoLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
              .addGroup(jPnTreinamentoLayout.createSequentialGroup()
                .addComponent(jLabel4)
                .addGap(0, 374, Short.MAX_VALUE))
              .addComponent(jTxtArquivoDadosTeste))
            .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
            .addComponent(jBtnAbrirDadosTeste, javax.swing.GroupLayout.PREFERRED_SIZE, 70, javax.swing.GroupLayout.PREFERRED_SIZE))
          .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPnTreinamentoLayout.createSequentialGroup()
            .addGap(0, 0, Short.MAX_VALUE)
            .addComponent(jBtnSalvarTeste)
            .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
            .addComponent(jBtnTestar, javax.swing.GroupLayout.PREFERRED_SIZE, 200, javax.swing.GroupLayout.PREFERRED_SIZE))
          .addGroup(jPnTreinamentoLayout.createSequentialGroup()
            .addComponent(jLabel6)
            .addGap(0, 0, Short.MAX_VALUE))
          .addComponent(jScrollPane3))
        .addContainerGap())
    );
    jPnTreinamentoLayout.setVerticalGroup(
      jPnTreinamentoLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
      .addGroup(jPnTreinamentoLayout.createSequentialGroup()
        .addContainerGap()
        .addComponent(jLabel6)
        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
        .addComponent(jLabel4)
        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
        .addGroup(jPnTreinamentoLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
          .addComponent(jTxtArquivoDadosTeste, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
          .addComponent(jBtnAbrirDadosTeste))
        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
        .addGroup(jPnTreinamentoLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
          .addComponent(jBtnTestar)
          .addComponent(jBtnSalvarTeste))
        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
        .addComponent(jScrollPane3, javax.swing.GroupLayout.PREFERRED_SIZE, 150, javax.swing.GroupLayout.PREFERRED_SIZE)
        .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
    );

    javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
    getContentPane().setLayout(layout);
    layout.setHorizontalGroup(
      layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
      .addGroup(layout.createSequentialGroup()
        .addContainerGap()
        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
          .addComponent(jPnAbrirArquivo, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
          .addComponent(jPnTreinamento, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        .addContainerGap())
    );
    layout.setVerticalGroup(
      layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
      .addGroup(layout.createSequentialGroup()
        .addContainerGap()
        .addComponent(jPnAbrirArquivo, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
        .addComponent(jPnTreinamento, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
        .addContainerGap())
    );

    pack();
  }// </editor-fold>//GEN-END:initComponents

  private void jBtnTreinarActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jBtnTreinarActionPerformed
    // TODO add your handling code here:
    acaoBotaoTreinar(false);
  }//GEN-LAST:event_jBtnTreinarActionPerformed

  private void jBtnAbrirDadosTreinamentoActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jBtnAbrirDadosTreinamentoActionPerformed
    abrirArquivoTreinamento();
  }//GEN-LAST:event_jBtnAbrirDadosTreinamentoActionPerformed

  private void jBtnAbrirDadosTesteActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jBtnAbrirDadosTesteActionPerformed
    // TODO add your handling code here:
    abrirArquivoTeste();
  }//GEN-LAST:event_jBtnAbrirDadosTesteActionPerformed

  private void jBtnTestarActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jBtnTestarActionPerformed
    // TODO add your handling code here:
    testarRede();
  }//GEN-LAST:event_jBtnTestarActionPerformed

  private void jBtnSalvarTreinamentoActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jBtnSalvarTreinamentoActionPerformed
    // TODO add your handling code here:
    salvarTreinamento();
  }//GEN-LAST:event_jBtnSalvarTreinamentoActionPerformed

  private void jBtnSalvarTesteActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jBtnSalvarTesteActionPerformed
    // TODO add your handling code here:
    salvarTeste();
  }//GEN-LAST:event_jBtnSalvarTesteActionPerformed

  private void jBtnTreinarMomentumActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jBtnTreinarMomentumActionPerformed
    // TODO add your handling code here:
    acaoBotaoTreinar(true);
  }//GEN-LAST:event_jBtnTreinarMomentumActionPerformed

  private void jBtnIniciarPesosActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jBtnIniciarPesosActionPerformed
    // TODO add your handling code here:
    iniciarPesos();
  }//GEN-LAST:event_jBtnIniciarPesosActionPerformed

  /**
   * @param args the command line arguments
   */
  public static void main(String args[]) {
    /* Set the Nimbus look and feel */
    //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
    /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
     */
    try {
      for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
        if ("Nimbus".equals(info.getName())) {
          javax.swing.UIManager.setLookAndFeel(info.getClassName());
          break;
        }
      }
    } catch (ClassNotFoundException ex) {
      java.util.logging.Logger.getLogger(Main.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
    } catch (InstantiationException ex) {
      java.util.logging.Logger.getLogger(Main.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
    } catch (IllegalAccessException ex) {
      java.util.logging.Logger.getLogger(Main.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
    } catch (javax.swing.UnsupportedLookAndFeelException ex) {
      java.util.logging.Logger.getLogger(Main.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
    }
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>
    //</editor-fold>

    /* Create and display the form */
    java.awt.EventQueue.invokeLater(new Runnable() {
      public void run() {
        new Main().setVisible(true);
      }
    });
  }

  // Variables declaration - do not modify//GEN-BEGIN:variables
  private javax.swing.JButton jBtnAbrirDadosTeste;
  private javax.swing.JButton jBtnAbrirDadosTreinamento;
  private javax.swing.JButton jBtnIniciarPesos;
  private javax.swing.JButton jBtnSalvarTeste;
  private javax.swing.JButton jBtnSalvarTreinamento;
  private javax.swing.JButton jBtnTestar;
  private javax.swing.JButton jBtnTreinar;
  private javax.swing.JButton jBtnTreinarMomentum;
  private javax.swing.JLabel jLabel1;
  private javax.swing.JLabel jLabel2;
  private javax.swing.JLabel jLabel3;
  private javax.swing.JLabel jLabel4;
  private javax.swing.JLabel jLabel6;
  private javax.swing.JPanel jPnAbrirArquivo;
  private javax.swing.JPanel jPnTreinamento;
  private javax.swing.JScrollPane jScrollPane1;
  private javax.swing.JScrollPane jScrollPane3;
  private javax.swing.JTextField jTxtArquivoDadosTeste;
  private javax.swing.JTextField jTxtArquivoDadosTreinamento;
  private javax.swing.JTextArea jTxtLogTeste;
  private javax.swing.JTextArea jTxtLogTreinamento;
  // End of variables declaration//GEN-END:variables

  private void abrirArquivoTreinamento() {
    String pasta;
    String nome;

    if (this.janelaAbrir == null) {
      this.janelaAbrir = new FileDialog((java.awt.Frame) null, "Selecionar arquivo", FileDialog.LOAD);
    }

    this.janelaAbrir.setMultipleMode(false);

    this.janelaAbrir.setVisible(true);

    pasta = janelaAbrir.getDirectory();
    nome = janelaAbrir.getFile();

    if (pasta != null && nome != null) {
      arquivoTreinamento = new Arquivo(pasta, nome);

      if (arquivoTreinamento.validarArquivo() == false) {
        JOptionPane.showMessageDialog(null, "Erro ao abrir o arquivo, ele está vazio ou não existe.", "Erro ao abrir arquivo", JOptionPane.ERROR_MESSAGE);
        return;
      }

      if (arquivoTreinamento.validarDados() == false) {
        JOptionPane.showMessageDialog(null, "Os dados de treinamento não estão no formato correto.", "Erro ao abrir arquivo", JOptionPane.ERROR_MESSAGE);
        return;
      }

      jTxtArquivoDadosTreinamento.setText(arquivoTreinamento.getCaminhoCompleto());
      jBtnIniciarPesos.setEnabled(true);
    }

  }

  private void abrirArquivoTeste() {
    String pasta;
    String nome;

    if (this.janelaAbrir == null) {
      this.janelaAbrir = new FileDialog((java.awt.Frame) null, "Selecionar arquivo", FileDialog.LOAD);
    }

    this.janelaAbrir.setMultipleMode(false);

    this.janelaAbrir.setVisible(true);

    pasta = janelaAbrir.getDirectory();
    nome = janelaAbrir.getFile();

    if (pasta != null && nome != null) {
      arquivoTeste = new Arquivo(pasta, nome);

      if (arquivoTreinamento.validarArquivo() == false) {
        JOptionPane.showMessageDialog(null, "Erro ao abrir o arquivo, ele está vazio ou não existe.", "Erro ao abrir arquivo", JOptionPane.ERROR_MESSAGE);
        return;
      }

      if (arquivoTreinamento.validarDados() == false) {
        JOptionPane.showMessageDialog(null, "Os dados de treinamento não estão no formato correto.", "Erro ao abrir arquivo", JOptionPane.ERROR_MESSAGE);
        return;
      }

      jTxtArquivoDadosTeste.setText(arquivoTeste.getCaminhoCompleto());
      jBtnTestar.setEnabled(true);
    }

  }

  private void acaoBotaoTreinar(boolean momentum) {
    if (arquivoTreinamento == null) {
      JOptionPane.showMessageDialog(null, "Selecione o arquivo com os dados de treinamento.", "Erro ao abrir arquivo", JOptionPane.ERROR_MESSAGE);
      return;
    }

    if (arquivoTreinamento.validarArquivo() == false) {
      JOptionPane.showMessageDialog(null, "Erro ao abrir o arquivo, ele está vazio ou não existe.", "Erro ao abrir arquivo", JOptionPane.ERROR_MESSAGE);
      return;
    }

    Comunicador.setCampo(jTxtLogTreinamento);
    Comunicador.setBotaoTestar(jBtnAbrirDadosTeste);
    Comunicador.setjBtnSalvar(jBtnSalvarTreinamento);
    Comunicador.setEnabledBotaoTestar(false);
    Comunicador.setEnabledBotaoSalvar(false);
    treinarRede(momentum);
  }

  private void treinarRede(boolean momentum) {
    ThreadAcao thread;

    thread = new ThreadAcao(redeMLP);
    thread.setArquivoTreinamento(arquivoTreinamento);

    if (!momentum) {
      thread.setTipoTreinamento(ThreadAcao.TREINAR_NORMAL);
    } else {
      thread.setTipoTreinamento(ThreadAcao.TREINAR_MOMENTUM);
    }

    thread.start();
  }

  private void testarRede() {
    if (arquivoTeste == null) {
      JOptionPane.showMessageDialog(null, "Selecione o arquivo com os dados de teste.", "Erro ao abrir arquivo", JOptionPane.ERROR_MESSAGE);
      return;
    }

    if (arquivoTeste.validarArquivo() == false) {
      JOptionPane.showMessageDialog(null, "Erro ao abrir o arquivo, ele está vazio ou não existe.", "Erro ao abrir arquivo", JOptionPane.ERROR_MESSAGE);
      return;
    }

    if (redeMLP != null) {
      Comunicador.setCampo(jTxtLogTeste);
      jBtnSalvarTeste.setEnabled(false);
      redeMLP.testar(arquivoTeste);
      jBtnSalvarTeste.setEnabled(true);
    }
  }

  private void salvarTreinamento() {
    salvarArquivo(jTxtLogTreinamento.getText());
  }

  private void salvarTeste() {
    salvarArquivo(jTxtLogTeste.getText());
  }

  private void salvarArquivo(String texto) {
    String nome;
    String pasta;
    Arquivo arquivoSalvar;

    if (this.janelaSalvar == null) {
      this.janelaSalvar = new FileDialog(this, "Salvar arquivo", FileDialog.SAVE);
    }

    this.janelaSalvar.setVisible(true);

    pasta = this.janelaSalvar.getDirectory();
    nome = this.janelaSalvar.getFile();

    if (pasta == null || nome == null) {
      return;
    }

    if (nome.indexOf(".") == -1) {
      nome += ".txt";
    }

    arquivoSalvar = new Arquivo(pasta, nome);

    if (arquivoSalvar.salvarArquivo(texto.replaceAll("\r", "\n"))) {
      JOptionPane.showMessageDialog(null, "Arquivo salvo com sucesso.", "Sucesso", JOptionPane.INFORMATION_MESSAGE);
    } else {
      JOptionPane.showMessageDialog(null, "Erro ao salvar arquivo.", "Erro", JOptionPane.ERROR_MESSAGE);
    }
  }

  private void iniciarPesos() {
    Comunicador.setCampo(jTxtLogTreinamento);
    this.redeMLP = new MLP();
    jBtnTreinar.setEnabled(true);
    jBtnTreinarMomentum.setEnabled(true);
    jBtnSalvarTreinamento.setEnabled(true);
  }
}
