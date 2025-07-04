import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { Chroma } from '@langchain/community/vectorstores/chroma';
import { ChromaClient } from 'chromadb';

const model = new ChatOpenAI({
	model: 'gpt-4o-2024-11-20',
	temperature: 0.8,
});

const embeddings = new OpenAIEmbeddings({
	model: 'text-embedding-3-small',
	openAIApiKey: process.env.OPENAI_API_KEY!,
});

const chromaClient = new ChromaClient({
	host: 'localhost',
	port: 5000,
});

//Create the collection program

const collectionName = 'ai_course-info';

async function createCollection() {
	await chromaClient.createCollection({ name: collectionName });
	console.log('Collection created successfully');
	const collections = await chromaClient.listCollections();
	console.log(collections);
}

const question = 'What will I learn at week 8 in this course?';
// const question = 'What themes does Gone with the Wind explore?';

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

	// metadata sanitize
	const sanitizedDocs = splittedDocs.map((doc, i) => ({
		pageContent: doc.pageContent,
		metadata: { id: `chunk_${i}` }, // ✅ valid metadata
	}));

	// Store the data
	const vectorStores = await Chroma.fromDocuments(sanitizedDocs, embeddings, {
		collectionName: collectionName,
		index: chromaClient,
	});

	console.log('Data entry successfully');

	// Manual embedding and query
	const embedded = await embeddings.embedQuery(question);
	console.log('✅ Embedding done, vector length:', embedded.length);

	const results = await vectorStores.similaritySearchVectorWithScore(embedded, 2);
	console.log('✅ Search done, results:', results.length);

	const resultDocs = results.map(([doc]) => doc.pageContent);
	console.log('🔍 Retrieved docs:', resultDocs);

	// //Build Chat template
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

// createCollection();
main();
