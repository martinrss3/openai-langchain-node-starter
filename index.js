import { OpenAI } from "langchain/llms";
import { RetrievalQAChain } from "langchain/chains";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import * as fs from "fs";
import * as dotenv from "dotenv";

dotenv.config();

const txtFileName = "Fausto-Goethe";
const question = "¿Quién es Mefistófeles?";
const txtPath = `./${txtFileName}.txt`;
const VECTOR_STORE_PATH = `./${txtFileName}.index`;

export const runWithEmbeddings = async () => {
  const model = new OpenAI({});

  let vectorStore;
  if (fs.existsSync(VECTOR_STORE_PATH)) {
    console.log("Vector Exists..");
    vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, new OpenAIEmbeddings());
  } else {
    const text = fs.readFileSync(txtPath, "utf8");
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
    });
    const docs = await textSplitter.createDocuments([text]);
    vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
    await vectorStore.save(VECTOR_STORE_PATH);
  }

  const chain = RetrievalQAChain.fromLLMS(model, vectorStore.asRetriever);

  const res = await chain.call({
    query: question,
  });

  console.log(res);
};

runWithEmbeddings();
