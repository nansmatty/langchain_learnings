import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';

const model = new ChatOpenAI({
	model: 'gpt-4o-2024-11-20',
	temperature: 0.8,
});

// Find the documentation here : https://v03.api.js.langchain.com

const question = 'What will I learn at week 7 in this course?';

async function main() {
	// Load the path of the pdf file;
	const loader = new PDFLoader('ai_misogi.pdf', {
		splitPages: false,
	});
	const docs = await loader.load();

	// Split the docs:
	const splitter = new RecursiveCharacterTextSplitter({
		separators: [`. \n`],
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
