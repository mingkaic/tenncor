#include "libgexf/libgexf.h"

#ifndef LIBGEXF_STREAMWRITER_HPP
#define LIBGEXF_STREAMWRITER_HPP

namespace libgexf
{

class StreamWriter {
public:
    /*! \var enum ElemType
     *  \brief Possible type of element
     */
    enum ElemType { NODE, EDGE };
public:
    StreamWriter() : _gexf(nullptr) {}

    /*!
     *  \brief Constructor with init
     *
     *  \param filepath : Path to the written file
     *  \param gexf : reference to a GEXF object
     */
    StreamWriter(GEXF* gexf) : _gexf(gexf) {}

    /*!
     *  \brief Copy constructor
     */
    StreamWriter(const StreamWriter& orig) : _gexf(orig._gexf) {}

    virtual ~StreamWriter() {}


    /*!
     *  \brief Get a duplicated instance of the internal GEXF data
     *
     *  \return GEXF instance
     */
    libgexf::GEXF getGEXFCopy() {
		return GEXF(*_gexf);
	}

    /*!
     *  \brief Initialize the file writer
     *
     *  \param gexf : reference to a GEXF object
     */
    void init(libgexf::GEXF* gexf)
	{
		_gexf = gexf;
	}

    /*!
     *  \brief Write to the outstream
     *
     */
    void write(std::ostream& os)
	{
		#ifndef LIBXML_READER_ENABLED
		throw FileWriterException("LIBXML NOT FOUND" );
		#endif

	 	/*
		 * this initialize the library and check potential ABI mismatches
		 * between the version it was compiled for and the actual shared
		 * library used.
		 */
		LIBXML_TEST_VERSION

		/* Create a new XmlWriter for _filepath, with no compression. */
		xmlBufferPtr buf = xmlBufferCreate();
		if (buf == NULL) {
			throw FileWriterException("Error creating the xml buffer");
		}
		xmlTextWriterPtr writer = xmlNewTextWriterMemory(buf, 0);
		if (writer == NULL) {
			throw FileWriterException("Error creating the xml FileWriter" );
		}

		/* Start the document with the xml default for the version,
		 * encoding _ENCODING and the default for the standalone
		 * declaration. */
		int rc = xmlTextWriterStartDocument(writer, NULL, _ENCODING, NULL);
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterStartDocument" );
		}

		this->writeGexfNode(writer);

		/* Here we could close the elements ORDER and EXAMPLE using the
		 * function xmlTextWriterEndElement, but since we do not want to
		 * write any other elements, we simply call xmlTextWriterEndDocument,
		 * which will do all the work. */
		rc = xmlTextWriterEndDocument(writer);
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterEndDocument" );
		}

		os << buf->content;

		/* Close file and free memory buffers */
		xmlFreeTextWriter(writer);
		xmlCleanupParser();
	}

private:
    void writeGexfNode(xmlTextWriterPtr writer){
		/* Start an element named "gexf". Since thist is the first
		 * element, this will be the root element of the document. */
		int rc = xmlTextWriterStartElement(writer, BAD_CAST "gexf");
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterStartElement" );
		}

		/* Start an element named "xmlns" */
		rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "xmlns",
			BAD_CAST _gexf->getMetaData().getXmlns().c_str());
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterWriteAttribute");
		}

		/* Start an element named "xsi" */
		rc = xmlTextWriterWriteAttributeNS(writer, BAD_CAST "xmlns",
			BAD_CAST "xsi", NULL,
			BAD_CAST _gexf->getMetaData().getXsi().c_str());
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterWriteAttributeNS");
		}

		/* Start an element named "schemaLocation" */
		rc = xmlTextWriterWriteAttributeNS(writer, BAD_CAST "xsi",
			BAD_CAST "schemaLocation", NULL,
			BAD_CAST _gexf->getMetaData().getSchema().c_str());
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterWriteAttributeNS");
		}

		/* Add an attribute with name "version" */
		rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "version",
			BAD_CAST _gexf->getMetaData().getVersion().c_str());
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterWriteAttribute");
		}

		this->writeMetaNode(writer);

		this->writeGraphNode(writer);

		/* Close the element named gexf. */
		rc = xmlTextWriterEndElement(writer);
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterEndElement" );
		}
	}
    void writeMetaNode(xmlTextWriterPtr writer){
		/* Do we have to write all these nodes? */
		const bool do_lastmodifieddate = _gexf->getMetaData().
			getLastModifiedDate().compare("") != 0;
		const bool do_creator = _gexf->getMetaData().
			getCreator().compare("") != 0;
		const bool do_desc = _gexf->getMetaData().
			getDescription().compare("") != 0;
		const bool do_kw = _gexf->getMetaData().getKeywords().compare("") != 0;
		const bool do_meta =
			do_creator || do_desc || do_kw || do_lastmodifieddate;

		if( !do_meta ) return;

		/* Start an element named "meta" as child of gexf. */
		int rc = xmlTextWriterStartElement(writer, BAD_CAST "meta");
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterStartElement");
		}

		/* Add an attribute with name "lastmodifieddate" */
		if( do_lastmodifieddate ) {
			rc = xmlTextWriterWriteAttribute(
				writer, BAD_CAST "lastmodifieddate",
				BAD_CAST _gexf->getMetaData().getLastModifiedDate().c_str());
			if (rc < 0) {
				throw FileWriterException(
					"Error at xmlTextWriterWriteAttribute");
			}
		}

		/* Write a text element named "creator" */
		if( do_creator ) {
			rc = xmlTextWriterWriteElement(writer, BAD_CAST "creator",
				BAD_CAST _gexf->getMetaData().getCreator().c_str());
			if (rc < 0) {
				throw FileWriterException("Error at xmlTextWriterWriteElement");
			}
		}

		/* Write a text element named "description" */
		if( do_desc ) {
			rc = xmlTextWriterWriteElement(writer, BAD_CAST "description",
				BAD_CAST _gexf->getMetaData().getDescription().c_str());
			if (rc < 0) {
				throw FileWriterException("Error at xmlTextWriterWriteElement");
			}
		}

		/* Write a text element named "keywords" */
		if( do_kw ) {
			rc = xmlTextWriterWriteElement(writer, BAD_CAST "keywords",
				BAD_CAST _gexf->getMetaData().getKeywords().c_str());
			if (rc < 0) {
				throw FileWriterException("Error at xmlTextWriterWriteElement");
			}
		}

		/* Close the element named meta. */
		rc = xmlTextWriterEndElement(writer);
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterEndElement");
		}
	}

    void writeGraphNode(xmlTextWriterPtr writer){
		/* Start an element named "graph" as child of gexf. */
		int rc = xmlTextWriterStartElement(writer, BAD_CAST "graph");
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterStartElement");
		}

		/* Add an attribute with name "mode" */
		rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "mode", BAD_CAST "static");
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterWriteAttribute");
		}

		/* Add an attribute with name "defaultedgetype" */
		t_graph t = _gexf->getGraphType();
		if (t == GRAPH_DIRECTED) {
			rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "defaultedgetype", BAD_CAST "directed");
		}
		else {
			rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "defaultedgetype", BAD_CAST "undirected");
		}
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterWriteAttribute");
		}

		//_gexf->getData().
		this->writeAttributesNode(writer,"node");
		this->writeAttributesNode(writer,"edge");
		this->writeNodesNode(writer);
		this->writeEdgesNode(writer);

		/* Close the element named graph. */
		rc = xmlTextWriterEndElement(writer);
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterEndElement");
		}
	}
    void writeNodesNode(xmlTextWriterPtr writer){
		/* Start an element named "nodes" as child of graph. */
		int rc = xmlTextWriterStartElement(writer, BAD_CAST "nodes");
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterStartElement");
		}

		/* Iterate on each node */
		NodeIter* it = _gexf->getUndirectedGraph().getNodes();
		while(it->hasNext()) {
			const t_id node_id = it->next();
			const std::string label = _gexf->getData().getNodeLabel(node_id);
			this->writeNodeNode(writer, Conv::idToStr(node_id), label);
		}

		/* Close the element named nodes. */
		rc = xmlTextWriterEndElement(writer);
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterEndElement");
		}
	}
    void writeNodeNode(xmlTextWriterPtr writer,
		const std::string& node_id, const std::string& label=""){
		/* Write an element named "node" as child of nodes. */
		int rc = xmlTextWriterStartElement(writer, BAD_CAST "node");
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterWriteElement");
		}

		/* Add an attribute with name "id" */
		rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "id", BAD_CAST node_id.c_str());
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterWriteAttribute");
		}

		/* Add an attribute with name "label" if necessary (optional as of gexf 1.2) */
		if(label.length() > 0) {
			rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "label", BAD_CAST label.c_str());
			if (rc < 0) {
				throw FileWriterException("Error at xmlTextWriterWriteAttribute");
			}
		}

		AttValueIter* row = _gexf->getData().getNodeAttributeRow(Conv::strToId(node_id));
		if( row != NULL && row->hasNext() ) {
			this->writeAttvaluesNode(writer, NODE, node_id);
		}

		/* Close the element named node. */
		rc = xmlTextWriterEndElement(writer);
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterEndElement");
		}
	}
    void writeEdgesNode(xmlTextWriterPtr writer){
		/* Start an element named "edges" as child of graph. */
		int rc = xmlTextWriterStartElement(writer, BAD_CAST "edges");
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterStartElement");
		}

		/* Iterate on each edge */
		EdgeIter* it = _gexf->getUndirectedGraph().getEdges();
		while(it->hasNext()) {
			const t_id edge_id = it->next();
			const t_id source_id = it->currentSource();
			const t_id target_id = it->currentTarget();
			const float weight = (float)it->currentProperty(EDGE_WEIGHT);
			const t_edge_type type = (t_edge_type)it->currentProperty(EDGE_TYPE);
			this->writeEdgeNode(writer, Conv::idToStr(edge_id), Conv::idToStr(source_id), Conv::idToStr(target_id), Conv::floatToStr(weight), Conv::edgeTypeToStr(type));
		}

		/* Close the element named edges. */
		rc = xmlTextWriterEndElement(writer);
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterEndElement");
		}
	}
    void writeEdgeNode(xmlTextWriterPtr writer, const std::string& edge_id,
		const std::string& source_id, const std::string& target_id,
		const std::string& weight="1", const std::string& type="undirected"){
		/* Write an element named "edge" as child of edges. */
		int rc = xmlTextWriterStartElement(writer, BAD_CAST "edge");
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterWriteElement");
		}

		/* Add an attribute with name "id" */
		rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "id", BAD_CAST edge_id.c_str());
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterWriteAttribute");
		}

		/* Add an attribute with name "source" */
		rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "source", BAD_CAST source_id.c_str());
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterWriteAttribute");
		}

		/* Add an attribute with name "target" */
		rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "target", BAD_CAST target_id.c_str());
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterWriteAttribute");
		}

		/* Add an attribute with name "weight" */
		if(weight.compare("1") > 0) {
			/* 1 is a defaultValue and can be omitted */
			rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "weight", BAD_CAST weight.c_str());
			if (rc < 0) {
				throw FileWriterException("Error at xmlTextWriterWriteAttribute");
			}
		}

		/* Add an attribute with name "type" */
		if( type.compare("undef") != 0 ) {
			t_graph t = _gexf->getGraphType();
			if( (t != GRAPH_DIRECTED && type.compare("undirected") != 0) || /* undirected is the default value and can be omitted */
				(t == GRAPH_DIRECTED && type.compare("directed") != 0) ) { /* directed can be omitted if it is the default value */

				rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "type", BAD_CAST type.c_str());
				if (rc < 0) {
					throw FileWriterException("Error at xmlTextWriterWriteAttribute");
				}
			}
		}

		AttValueIter* row = _gexf->getData().getEdgeAttributeRow(Conv::strToId(edge_id));
		if( row != NULL && row->hasNext() ) {
			this->writeAttvaluesNode(writer, EDGE, edge_id);
		}

		/* Close the element named edge. */
		rc = xmlTextWriterEndElement(writer);
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterEndElement");
		}
	}

    void writeAttributesNode(xmlTextWriterPtr writer,
		const std::string& element_class){
		AttributeIter* it = 0;

		if( element_class.compare("node") != 0 && element_class.compare("edge") != 0 ) {
			throw std::invalid_argument("writeAttributesNode: invalid element_class");
		}

		if(element_class.compare("node") == 0) {
			it = _gexf->getData().getNodeAttributeColumn();
		}
		else if(element_class.compare("edge") == 0) {
			it = _gexf->getData().getEdgeAttributeColumn();
		}
		if( !it->hasNext() ) return;

		/* Start an element named "attributes" as child of graph. */
		int rc = xmlTextWriterStartElement(writer, BAD_CAST "attributes");
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterStartElement");
		}

		/* Add an attribute with name "class" */
		rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "class", BAD_CAST element_class.c_str());
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterWriteAttribute");
		}

		/* Add an attribute with name "mode" */
		rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "mode", BAD_CAST "static");
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterWriteAttribute");
		}

		while(it->hasNext()) {
			const t_id attr_id = it->next();
			const std::string title = it->currentTitle();
			const t_attr_type type = it->currentType();
			std::string default_value = "";
			std::string options = "";

			if(element_class.compare("node") == 0) {
				if( _gexf->getData().hasNodeAttributeDefault(attr_id) ) {
					default_value = _gexf->getData().getNodeAttributeDefault(attr_id);
					options = _gexf->getData().getNodeAttributeOptions(attr_id);
				}
			}
			else if(element_class.compare("edge") == 0) {
				if( _gexf->getData().hasEdgeAttributeDefault(attr_id) ) {
					default_value = _gexf->getData().getEdgeAttributeDefault(attr_id);
					options = _gexf->getData().getEdgeAttributeOptions(attr_id);
				}
			}
			this->writeAttributeNode(writer, Conv::idToStr(attr_id), title, Conv::attrTypeToStr(type), default_value, options);
		}

		/* Close the element named attributes. */
		rc = xmlTextWriterEndElement(writer);
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterEndElement");
		}
	}
    void writeAttributeNode(xmlTextWriterPtr writer,
		const std::string& id, const std::string& title,
		const std::string& type,
		const std::string& default_value="", const std::string& options=""){
		/* Start an element named "attribute" as child of attributes. */
		int rc = xmlTextWriterStartElement(writer, BAD_CAST "attribute");
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterStartElement");
		}

		/* Add an attribute with name "id" */
		rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "id", BAD_CAST id.c_str());
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterWriteAttribute");
		}

		/* Add an attribute with name "title" */
		rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "title", BAD_CAST title.c_str());
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterWriteAttribute");
		}

		/* Add an attribute with name "type" */
		rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "type", BAD_CAST type.c_str());
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterWriteAttribute");
		}

		/* Write a text element named "default" */
		if( !default_value.empty() ) {
			rc = xmlTextWriterWriteElement(writer, BAD_CAST "default", BAD_CAST default_value.c_str());
			if (rc < 0) {
				throw FileWriterException("Error at xmlTextWriterWriteElement");
			}
		}

		/* Write a text element named "options" */
		if( !options.empty() ) {
			rc = xmlTextWriterWriteElement(writer, BAD_CAST "options", BAD_CAST options.c_str());
			if (rc < 0) {
				throw FileWriterException("Error at xmlTextWriterWriteElement");
			}
		}

		/* Close the element named attribute. */
		rc = xmlTextWriterEndElement(writer);
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterEndElement");
		}
	}
    void writeAttributeDefaultNode(xmlTextWriterPtr writer,
		const std::string& default_value){
		/* Start an element named "default" as child of attribute. */
		int rc = xmlTextWriterStartElement(writer, BAD_CAST "default");
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterStartElement");
		}

		/* Write a text element named "default" */
		rc = xmlTextWriterWriteElement(writer, BAD_CAST "default", BAD_CAST default_value.c_str());
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterWriteElement");
		}

		/* Close the element named default. */
		rc = xmlTextWriterEndElement(writer);
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterEndElement");
		}
	}
    void writeAttvaluesNode(xmlTextWriterPtr writer,
		const ElemType type, const std::string& id){
		/* Start an element named "attvalues" as child of node or edge. */
		int rc = xmlTextWriterStartElement(writer, BAD_CAST "attvalues");
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterStartElement");
		}

		/* Write each attribute row */
		AttValueIter* row = 0;
		if( type == NODE ) {
			row = _gexf->getData().getNodeAttributeRow(Conv::strToId(id));
		}
		else if( type == EDGE ) {
			row = _gexf->getData().getEdgeAttributeRow(Conv::strToId(id));
		}
		if(row != NULL) {
			while(row->hasNext()) {
				const t_id attr_id = row->next();
				const std::string v = row->currentValue();
				this->writeAttvalueNode(writer, Conv::idToStr(attr_id), v);
			}
		}

		/* Close the element named default. */
		rc = xmlTextWriterEndElement(writer);
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterEndElement");
		}
	}
    void writeAttvalueNode(xmlTextWriterPtr writer,
		const std::string& attribute_id, const std::string& value){
		/* Start an element named "attvalue" as child of attvalues. */
		int rc = xmlTextWriterStartElement(writer, BAD_CAST "attvalue");
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterStartElement");
		}

		/* Add an attribute with name "for" */
		rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "for", BAD_CAST attribute_id.c_str());
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterWriteAttribute");
		}

		/* Add an attribute with name "value" */
		rc = xmlTextWriterWriteAttribute(writer, BAD_CAST "value", BAD_CAST value.c_str());
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterWriteAttribute");
		}

		/* Close the element named attvalue. */
		rc = xmlTextWriterEndElement(writer);
		if (rc < 0) {
			throw FileWriterException("Error at xmlTextWriterEndElement");
		}
	}

private:
    GEXF* _gexf;
    static const char* _ENCODING;
};

}

#endif // LIBGEXF_STREAMWRITER_HPP
