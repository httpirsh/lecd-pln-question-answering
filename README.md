# lecd-pln-question-answering
Este projeto consiste na aplicação e exploração dos conteúdos lecionados em dois tipos de abordagem de **Question Answering**, sendo uma baseada em regras e outra em métodos de Aprendizagem Computacional.

**Ficheiros Presentes:**

**Dataset utilizado (input)**
 - dev.json (DREAM (https://github.com/nlpdata/dream))

Este dataset consiste numa lista de listas, onde cada lista interna contém informações relacionadas a uma conversa e uma pergunta e correspondente resposta, com as seguintes variáveis:
"question": Uma string que representa uma pergunta relacionada à conversa.
"choice": Uma lista de opções de resposta de escolha múltipla para a pergunta.
"answer": Uma string que indica a resposta correta entre as opções.

**Script (código)**
 - meta1.py

Este script tem como objetivo desenvolver um sistema de Question Answering, baseado em tarefas relacionadas ao Processamento de Linguagem Natural. Estas regras incluem a extração de bigramas e entidades, análise de sentimentos, previsão de respostas e avaliação do desempenho do modelo.
Para o dado dataset, seria uma aplicação da resposta mais acertada a perguntas baseadas em diálogo.

**Set-up e potenciais soluções para instalação de imports**

Para poder correr o código sem quaisquer constrangimentos é importante instalar previamente todos os imports necessários.
Como o método de avaliação métrica SAS não está incluído na biblioteca sklearn, é necessário instalar o devidos imports através dos comandos:
 > pip install torch

 > pip install transformers

Se tiver problemas ao instalar o módulo transformers, poderá ser porque o seu sistema não tem suporte para Windows Long Path habilitado. Nesse caso, instroduza a seguinte instrução na powershell windows(Admin):

 > New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
-Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

