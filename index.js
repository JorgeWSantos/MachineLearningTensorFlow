import * as tf from "@tensorflow/tfjs";

async function trainModel(inputXs, outputYs) {
  // Criamos o modelo sequencial
  const model = tf.sequential();

  //Primeira camada da rede neural
  // inputShape: [7] porque temos 7 entradas (idade, cor e localização)
  model.add(
    tf.layers.dense({
      inputShape: [7],
      // 80 neurônios na primeira camada, quanto mais neurônios,
      // mais complexa a rede, mas também mais lenta
      units: 80,
      // Função de ativação ReLU (Rectified Linear Unit) age como filtro,
      // permitindo apenas valores positivos passarem para a próxima camada
      // se a informação for negativa, ela é descartada, o que ajuda a rede a aprender padrões mais complexos
      activation: "relu",
    }),
  );

  // Segunda camada da rede neural
  model.add(
    tf.layers.dense({
      // 3 neurônios na segunda camada, pois temos 3 categorias de saída (premium, medium, basic)
      units: 3,
      //softmax é uma função de ativação que transforma os valores de saída em probabilidades, garantindo que a soma das saídas seja igual a 1
      activation: "softmax",
    }),
  );

  model.compile({
    // Otimizador Adam é um algoritmo de otimização que ajusta os pesos da rede neural durante o treinamento para minimizar a função de perda
    // Aprende com acertos e erros, ajustando os pesos da rede para melhorar a precisão das previsões
    optimizer: "adam",
    // Função de perda para classificação multiclasse, que mede a diferença entre as previsões do modelo e os rótulos reais,
    // mede os scores de probabilidade previstos pelo modelo em relação aos rótulos reais, penalizando mais fortemente as previsões incorretas
    // a categoria premium sempre será [1, 0, 0], medium será [0, 1, 0] e basic será [0, 0, 1]
    loss: "categoricalCrossentropy",
    // Métrica para avaliar o desempenho do modelo
    metrics: ["accuracy"],
  });

  // Treinamos o modelo com os dados de entrada (inputXs) e saída (outputYs)
  await model.fit(inputXs, outputYs, {
    epochs: 100, // Número de vezes que o modelo verá todo o conjunto de dados durante o treinamento
    shuffle: true, // Embaralha os dados a cada época para evitar padrões de aprendizado indesejados
    verbose: 0, // 0 = sem logs, 1 = logs de progresso, 2 = logs detalhados
    batchSize: 1, // Número de amostras processadas antes de atualizar os pesos do modelo
    // callbacks: {
    //   onEpochEnd: (epoch, logs) => {
    //     console.log(
    //       `epochs ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`,
    //     );
    //   },
    // },
  });

  return model;
}

async function predict(model, pessoaNormalizada) {
  // Fazemos a previsão com o modelo treinado
  const prediction = model.predict(tf.tensor2d(pessoaNormalizada));
  const predictionArray = prediction.arraySync();

  return predictionArray[0].map((prob, index) => ({
    prob: prob,
    index: index,
  }));
}

//////////==================
// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
  [0.33, 1, 0, 0, 1, 0, 0], // Erick
  [0, 0, 1, 0, 0, 1, 0], // Ana
  [1, 0, 0, 1, 0, 0, 1], // Carlos
];

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
  [1, 0, 0], // premium - Erick
  [0, 1, 0], // medium - Ana
  [0, 0, 1], // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado);
const outputYs = tf.tensor2d(tensorLabels);

// inputXs.print();
// outputYs.print();

const model = await trainModel(inputXs, outputYs);

// const pessoa = {
//   nome: "João",
//   cor: "verde",
//   localizacao: "Curitiba",
//   idade: 29,
// };

//normalizamos os dados de entrada da pessoa para que fiquem no mesmo formato do dataset de treino
// exemplo idade_min: 25, idade_max: 40, então, (29-25)/(40-25) = 0.26666666666666666
const pessoaNormalizada = [
  [
    0.26666666666666666, //idade normalizada
    1, // azul
    0, // vermelho
    0, // verde
    0, // São Paulo
    1, // Rio
    0, // Curitiba
  ],
];

const predictions = await predict(model, pessoaNormalizada);
const results = predictions
  .sort((a, b) => b.prob - a.prob)
  .map((p) => ({
    label: labelsNomes[p.index],
    probability: (p.prob * 100).toFixed(2) + "%", // Arredondamos para 4 casas decimais
  }));

console.log(results);
