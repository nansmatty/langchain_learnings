import { Document } from '@langchain/core/documents';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';

const model = new ChatOpenAI({
	model: 'gpt-4o-2024-11-20',
	temperature: 0.8,
});

const question = 'What is langchain? What are langchain libraries?';

async function main() {
	// Load the path url data

	const loader = new CheerioWebBaseLoader('https://js.langchain.com/docs/introduction/');
	const docs = await loader.load();

	// Split the docs:
	const splitter = new RecursiveCharacterTextSplitter({
		chunkSize: 200,
		chunkOverlap: 20,
	});

	const splittedDocs = await splitter.splitDocuments(docs);

	// Store the data
	const vectorStores = new MemoryVectorStore(new OpenAIEmbeddings());
	await vectorStores.addDocuments(splittedDocs);

	//Create a data retrival
	const retriver = vectorStores.asRetriever({
		k: 2,
	});

	//Get relevant documents
	const results = await retriver._getRelevantDocuments(question);

	const resultDocs = results.map((result) => result.pageContent);

	//Build Chat template
	const template = ChatPromptTemplate.fromMessages([
		['system', 'You are a helpful assistant.'],
		['system', 'Answer the question based on the following context: {context}'],
		['human', 'Question: {question}'],
	]);

	const chain = template.pipe(model);
	const response = await chain.invoke({
		context: resultDocs,
		question: question,
	});

	console.log(response.content);
}

main();
