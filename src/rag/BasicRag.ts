import { Document } from '@langchain/core/documents';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

const model = new ChatOpenAI({
	model: 'gpt-4o-2024-11-20',
	temperature: 0.8,
});

const myData = [
	'Dream big, but work even harder.',
	'Silence often says more than words.',
	'Kindness costs nothing but means everything.',
	'You become what you feed your mind.',
	'Time waits for no one—start now.',
	'Fear is temporary, regret is forever.',
	'Success whispers, not shouts.',
	'Every setback is a setup for a comeback.',
	'Growth begins where comfort ends.',
	'Life’s short—say what you feel.',
	'Hustle in silence, let your success speak.',
	'Chaos often precedes transformation.',
	'A calm mind is a powerful weapon.',
	'Don’t chase people—build yourself.',
	'You can’t pour from an empty cup.',
];

const question = 'How do I deal with setbacks in life?';

async function main() {
	//Step: 1 Store the data
	const vectorStores = new MemoryVectorStore(new OpenAIEmbeddings());
	await vectorStores.addDocuments(myData.map((content) => new Document({ pageContent: content })));

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
